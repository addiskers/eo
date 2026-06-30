import asyncio
import base64
import csv
import io
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive
from plivo_handler import PlivoMediaBridge

import dialer
import pricing
import scheduler
import store
from recorder import CallRecorder

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")
PLIVO_FROM_NUMBER = os.getenv("PLIVO_FROM_NUMBER", "")
ANALYTICS_SECRET = os.getenv("ANALYTICS_SECRET", "eo2026")

# ============ EVENT / GUEST DATA ============

EVENT = {
    "host": "EO Gujarat",
    "occasion": "Inaugural evening with Varun Dhawan",
    "date": "10th July",
    "city": "Ahmedabad",
}

def handle_record_rsvp(**kwargs):
    """The agent calls this once per call with the RSVP outcome."""
    status = (kwargs.get("outcome_status") or "").strip().lower()
    if status not in ("yes", "no", "callback", "do_not_contact"):
        # Back-compat: derive from the legacy `attending` boolean.
        status = "yes" if kwargs.get("attending") else "no"
    return {
        "success": True,
        "outcome_status": status,
        "attending": status == "yes",
        "callback_time_text": kwargs.get("callback_time_text", "") or "",
        "callback_time_iso": kwargs.get("callback_time_iso", "") or "",
        "do_not_contact": status == "do_not_contact",
        "accompanying_children": kwargs.get("accompanying_children", "") or "",
        "guest_name": kwargs.get("guest_name", "") or "",
        "note": kwargs.get("note", "") or "",
        "event": EVENT,
    }


# Live transcript watchers (browser WebSockets watching phone calls)
live_watchers: set = set()

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
async def _startup():
    """Initialize the call store, recover orphans, and start the callback scheduler."""
    try:
        await store.init()
        await store.sweep_stale()
        await store.reset_orphaned_callbacks()
    except Exception as e:
        logger.error(f"Call store init failed: {e}")
    try:
        app.state.callback_task = asyncio.create_task(scheduler.run_loop())
    except Exception as e:
        logger.error(f"Failed to start callback scheduler: {e}")


@app.on_event("shutdown")
async def _shutdown():
    task = getattr(app.state, "callback_task", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Gemini Live."""
    await websocket.accept()

    logger.info("WebSocket connection accepted")

    recorder = CallRecorder(model=MODEL)
    await recorder.open(source="browser")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()

    client_disconnected = False

    async def audio_output_callback(data):
        if not client_disconnected:
            try:
                await websocket.send_bytes(data)
            except Exception:
                pass

    async def audio_interrupt_callback():
        pass

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=MODEL,
        input_sample_rate=16000,
        tool_mapping={
            "record_rsvp": handle_record_rsvp,
        }
    )

    session_task = None

    async def receive_from_client():
        nonlocal client_disconnected
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")
        finally:
            client_disconnected = True
            if session_task and not session_task.done():
                session_task.cancel()

    receive_task = asyncio.create_task(receive_from_client())

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]

    async def run_session_with_retry():
        for attempt in range(MAX_RETRIES + 1):
            should_retry = False
            try:
                async for event in gemini_client.start_session(
                    audio_input_queue=audio_input_queue,
                    video_input_queue=video_input_queue,
                    text_input_queue=text_input_queue,
                    audio_output_callback=audio_output_callback,
                    audio_interrupt_callback=audio_interrupt_callback,
                ):
                    if event:
                        if event.get("type") == "error" and attempt < MAX_RETRIES:
                            error_msg = event.get("error", "")
                            if "exhausted" in error_msg or "quota" in error_msg.lower():
                                delay = RETRY_DELAYS[attempt]
                                logger.warning(f"Quota error, retrying in {delay}s (attempt {attempt+1}/{MAX_RETRIES})")
                                try:
                                    await websocket.send_json({"type": "status", "text": "Reconnecting..."})
                                except RuntimeError:
                                    return
                                await asyncio.sleep(delay)
                                should_retry = True
                                break
                        if event.get("type") == "go_away" and attempt < MAX_RETRIES:
                            logger.info(f"GoAway received, reconnecting (attempt {attempt+1}/{MAX_RETRIES})")
                            try:
                                await websocket.send_json({"type": "status", "text": "Reconnecting..."})
                            except RuntimeError:
                                return
                            await asyncio.sleep(1)
                            should_retry = True
                            break
                        await recorder.on_event(event)
                        try:
                            await websocket.send_json(event)
                        except RuntimeError:
                            return
                if not should_retry:
                    return
            except Exception as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Session error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

    try:
        session_task = asyncio.create_task(run_session_with_retry())
        await session_task
    except asyncio.CancelledError:
        logger.info("Gemini session cancelled due to client disconnect")
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        await recorder.close()
        try:
            await websocket.close()
        except:
            pass
        logger.info("connection closed")


# ============ PLIVO VOICE ENDPOINTS ============

@app.api_route("/plivo/answer", methods=["GET", "POST"])
async def plivo_answer(request: Request):
    """Plivo answer webhook: returns streaming XML when the callee picks up."""
    # Prefer PUBLIC_URL (most reliable behind tunnels/proxies); else infer from the
    # request, honouring X-Forwarded-Proto so we still pick wss behind nginx/ngrok.
    public_url = os.getenv("PUBLIC_URL", "").rstrip("/")
    if public_url:
        parsed = urlparse(public_url)
        host = parsed.netloc
        secure = parsed.scheme == "https"
    else:
        host = request.headers.get("host", "localhost")
        xf_proto = request.headers.get("x-forwarded-proto", "")
        secure = (request.url.scheme == "https" or xf_proto == "https"
                  or "onrender.com" in host or "globalvoxinc.ai" in host)
    ws_url = f"{'wss' if secure else 'ws'}://{host}/plivo/media-stream"

    caller = request.query_params.get("caller", "")
    gen = request.query_params.get("gen", "")
    origin = request.query_params.get("origin", "")
    hdr_pairs = []
    if caller:
        hdr_pairs.append(f"X-Caller={quote(caller)}")
    if gen:
        hdr_pairs.append(f"X-Callback-Gen={quote(gen)}")
    if origin:
        hdr_pairs.append(f"X-Callback-Origin={quote(origin)}")
    extra_headers = ",".join(hdr_pairs)
    eh_attr = f' extraHeaders="{extra_headers}"' if extra_headers else ""

    # Plivo <Stream>: URL is the element TEXT (not a url= attr); audioTrack must be
    # "inbound" with bidirectional="true"; codec mulaw 8kHz.
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        '  <Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000" audioTrack="inbound"{eh_attr}>'
        f'{ws_url}</Stream>\n'
        '</Response>'
    )
    return Response(content=xml, media_type="application/xml")


@app.websocket("/plivo/media-stream")
async def plivo_media_stream(websocket: WebSocket):
    """WebSocket endpoint for Plivo bidirectional Audio Streaming."""
    await websocket.accept()
    logger.info("Plivo Media Stream WebSocket accepted")

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY,
        model=MODEL,
        input_sample_rate=16000,
        tool_mapping={
            "record_rsvp": handle_record_rsvp,
        }
    )

    recorder = CallRecorder(model=MODEL)

    async def broadcast_event(event):
        """Send transcript events to all live watchers AND record the call."""
        # Persist (never let a recorder failure affect the live broadcast).
        etype = event.get("type")
        if etype == "call_start":
            await recorder.open(source="plivo", call_sid=event.get("call_sid") or None,
                                caller=event.get("caller"),
                                generation=event.get("generation", 0))
        elif etype == "call_end":
            await recorder.close()
        else:
            await recorder.on_event(event)

        dead = set()
        for watcher in live_watchers:
            try:
                await watcher.send_json(event)
            except Exception:
                dead.add(watcher)
        live_watchers.difference_update(dead)

    bridge = PlivoMediaBridge(
        websocket=websocket,
        gemini_client=gemini_client,
        text_trigger="[The guest has just answered the call. Greet them now with your invitation.]",
        on_event=broadcast_event,
    )

    try:
        await bridge.run()
    except Exception as e:
        import traceback
        logger.error(f"Plivo bridge error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.post("/call-me")
async def call_me(request: Request):
    """Make Plivo call a phone number and connect to the AI agent."""
    body = await request.json()
    to_number = body.get("phone")
    if not to_number:
        return {"error": "Missing 'phone' field. Send {\"phone\": \"+91XXXXXXXXXX\"}"}
    return await dialer.place_call(to_number, request=request)


# ============ LIVE TRANSCRIPT DASHBOARD ============

@app.get("/live")
async def live_dashboard():
    """Live transcript dashboard — watch phone calls in real-time."""
    return HTMLResponse(LIVE_DASHBOARD_HTML)


@app.websocket("/live/ws")
async def live_ws(websocket: WebSocket):
    """WebSocket for live transcript watchers."""
    await websocket.accept()
    live_watchers.add(websocket)
    logger.info(f"Live watcher connected ({len(live_watchers)} total)")
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except:
        pass
    finally:
        live_watchers.discard(websocket)
        logger.info(f"Live watcher disconnected ({len(live_watchers)} total)")


LIVE_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EO Gujarat · Live Transcript</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0e17;
  --card: rgba(17,24,39,0.75);
  --border: rgba(255,255,255,0.08);
  --cyan: #00d4ff;
  --purple: #7c3aed;
  --green: #10b981;
  --red: #ef4444;
  --text: #f1f5f9;
  --muted: #64748b;
  --secondary: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: rgba(10,14,23,0.9);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}
.brand {
  font-weight: 700;
  font-size: 0.9rem;
  color: var(--cyan);
}
.status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  font-weight: 600;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--muted);
}
.dot.live {
  background: var(--green);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.4); }
  50% { opacity:0.7; box-shadow: 0 0 0 4px rgba(16,185,129,0); }
}
.container {
  flex: 1;
  max-width: 700px;
  width: 100%;
  margin: 0 auto;
  padding: 20px;
  position: relative;
  z-index: 1;
}
.waiting {
  text-align: center;
  padding: 60px 20px;
  color: var(--muted);
}
.waiting h2 { font-size: 1.1rem; margin-bottom: 8px; color: var(--secondary); }
.waiting p { font-size: 0.8rem; }
#transcript {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.msg {
  padding: 10px 14px;
  border-radius: 12px;
  max-width: 85%;
  font-size: 0.875rem;
  line-height: 1.5;
  animation: fadeIn 0.2s ease-out;
}
@keyframes fadeIn {
  from { opacity:0; transform: translateY(8px); }
  to { opacity:1; transform: translateY(0); }
}
.msg .time {
  display: block;
  font-size: 0.6rem;
  opacity: 0.5;
  font-family: 'SF Mono', monospace;
  margin-top: 3px;
}
.msg.user {
  align-self: flex-end;
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,212,255,0.1));
  border: 1px solid rgba(0,212,255,0.15);
  border-bottom-right-radius: 4px;
}
.msg.gemini {
  align-self: flex-start;
  background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(124,58,237,0.1));
  border: 1px solid rgba(124,58,237,0.15);
  border-bottom-left-radius: 4px;
}
.msg.system {
  align-self: center;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.75rem;
  max-width: 100%;
  text-align: center;
}
.tool-card {
  align-self: center;
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.2);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.75rem;
  color: var(--green);
  max-width: 100%;
  animation: fadeIn 0.2s ease-out;
}
.tool-card .tool-name { font-weight: 700; }
.tool-card pre {
  margin-top: 6px;
  color: var(--secondary);
  font-size: 0.7rem;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
</head>
<body>
<div class="top-bar">
  <span class="brand">EO Gujarat · Live Transcript</span>
  <div class="status">
    <span class="dot" id="statusDot"></span>
    <span id="statusText">Waiting for call...</span>
  </div>
</div>
<div class="container">
  <div class="waiting" id="waiting">
    <h2>No active call</h2>
    <p>Start a call using the "Call Me" button or dial +1 (978) 571-5824.<br>The transcript will appear here in real-time.</p>
  </div>
  <div id="transcript"></div>
</div>
<script>
const transcript = document.getElementById('transcript');
const waiting = document.getElementById('waiting');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
let currentUser = null;
let currentGemini = null;

const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(protocol + '//' + location.host + '/live/ws');

ws.onopen = () => { statusText.textContent = 'Connected — waiting for call...'; };
ws.onclose = () => { statusText.textContent = 'Disconnected'; statusDot.className = 'dot'; };

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);

  if (msg.type === 'call_start') {
    waiting.style.display = 'none';
    statusDot.className = 'dot live';
    statusText.textContent = 'Call in progress';
    addSystem('Call started');
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'call_end') {
    statusDot.className = 'dot';
    statusText.textContent = 'Call ended';
    addSystem('Call ended');
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'user') {
    if (currentUser) {
      currentUser.querySelector('.text').textContent += msg.text;
    } else {
      currentUser = addMsg('user', msg.text);
      currentGemini = null;
    }
  }
  else if (msg.type === 'gemini') {
    if (currentGemini) {
      currentGemini.querySelector('.text').textContent += msg.text;
    } else {
      currentGemini = addMsg('gemini', msg.text);
      currentUser = null;
    }
  }
  else if (msg.type === 'turn_complete') {
    currentUser = null;
    currentGemini = null;
  }
  else if (msg.type === 'tool_call') {
    addTool(msg.name, msg.result);
  }

  window.scrollTo(0, document.body.scrollHeight);
};

function addMsg(type, text) {
  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const div = document.createElement('div');
  div.className = 'msg ' + type;
  div.innerHTML = '<span class="text"></span><span class="time">' + time + '</span>';
  div.querySelector('.text').textContent = text;
  transcript.appendChild(div);
  return div;
}

function addSystem(text) {
  const div = document.createElement('div');
  div.className = 'msg system';
  div.textContent = text;
  transcript.appendChild(div);
}

function addTool(name, result) {
  const div = document.createElement('div');
  div.className = 'tool-card';
  div.innerHTML = '<span class="tool-name">' + name + '</span><pre>' +
    JSON.stringify(result, null, 2).slice(0, 500) + '</pre>';
  transcript.appendChild(div);
}
</script>
</body>
</html>"""


ADMIN_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Admin · Call Analytics</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0a0e17; --card:rgba(17,24,39,0.75); --border:rgba(255,255,255,0.08);
  --cyan:#00d4ff; --purple:#7c3aed; --green:#10b981; --red:#ef4444; --amber:#f59e0b;
  --text:#f1f5f9; --muted:#64748b; --secondary:#94a3b8;
  --mono:'SF Mono',ui-monospace,Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:
  linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);
  background-size:40px 40px;pointer-events:none;z-index:0;}
.hidden{display:none !important;}
a{color:var(--cyan);}

/* ---- login ---- */
.login-wrap{position:relative;z-index:1;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}
.login-card{background:var(--card);backdrop-filter:blur(14px);border:1px solid var(--border);border-radius:18px;padding:34px 30px;width:100%;max-width:380px;text-align:center;}
.login-card h1{font-size:1.15rem;margin-bottom:6px;}
.login-card p{color:var(--muted);font-size:0.8rem;margin-bottom:20px;}
input,select{font-family:inherit;background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:10px;color:var(--text);padding:10px 12px;font-size:0.85rem;width:100%;outline:none;}
input:focus,select:focus{border-color:var(--cyan);}
.btn{font-family:inherit;cursor:pointer;border:none;border-radius:10px;padding:10px 16px;font-size:0.82rem;font-weight:600;color:#04121a;background:var(--cyan);transition:opacity .15s;}
.btn:hover{opacity:.88;}
.btn.ghost{background:transparent;border:1px solid var(--border);color:var(--secondary);}
.login-card .btn{width:100%;margin-top:14px;}
.err{color:var(--red);font-size:0.75rem;margin-top:10px;min-height:1em;}

/* ---- shell ---- */
.top-bar{position:sticky;top:0;z-index:10;display:flex;justify-content:space-between;align-items:center;
  padding:12px 22px;background:rgba(10,14,23,0.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);}
.brand{font-weight:800;font-size:0.92rem;color:var(--cyan);letter-spacing:.2px;}
.top-actions{display:flex;align-items:center;gap:10px;}
.status{display:flex;align-items:center;gap:6px;font-size:0.72rem;color:var(--secondary);}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2.4s infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,185,129,.4);}50%{opacity:.6;box-shadow:0 0 0 4px rgba(16,185,129,0);}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
@keyframes shimmer{0%{background-position:-400px 0;}100%{background-position:400px 0;}}
.container{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:22px;}
.section-title{font-size:0.7rem;text-transform:uppercase;letter-spacing:1.4px;color:var(--muted);margin:26px 4px 12px;}

/* ---- stat cards ---- */
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;animation:fadeIn .25s;}
.stat .label{font-size:0.66rem;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);}
.stat .value{font-size:1.5rem;font-weight:800;margin-top:8px;font-family:var(--mono);}
.stat .sub{font-size:0.68rem;color:var(--secondary);margin-top:4px;}
.stat.hi .value{color:var(--purple);}
.stat.cy .value{color:var(--cyan);}
.stat.gr .value{color:var(--green);}

/* ---- chart + breakdown ---- */
.row2{display:grid;grid-template-columns:2fr 1fr;gap:14px;}
.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;}
.panel h3{font-size:0.8rem;font-weight:600;margin-bottom:14px;color:var(--secondary);}
#trend svg{width:100%;height:220px;display:block;}
.brk-row{display:flex;align-items:center;justify-content:space-between;font-size:0.78rem;padding:7px 0;border-bottom:1px solid var(--border);}
.brk-row:last-child{border-bottom:none;}
.brk-bar{height:6px;border-radius:3px;background:var(--cyan);margin-top:5px;}

/* ---- filters + table ---- */
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px;}
.filters input,.filters select{width:auto;}
.filters .grow{flex:1;min-width:160px;}
.table-scroll{overflow-x:auto;background:var(--card);border:1px solid var(--border);border-radius:14px;}
table{width:100%;border-collapse:collapse;font-size:0.78rem;min-width:880px;}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--border);white-space:nowrap;}
th{font-size:0.66rem;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);cursor:pointer;user-select:none;position:sticky;top:0;background:#0d1320;}
th.num,td.num{text-align:right;font-family:var(--mono);}
tbody tr{cursor:pointer;transition:background .12s;}
tbody tr:hover{background:rgba(0,212,255,0.05);}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:0.66rem;font-weight:600;}
.pill.completed{background:rgba(16,185,129,.15);color:var(--green);}
.pill.in_progress{background:rgba(0,212,255,.15);color:var(--cyan);}
.pill.abandoned,.pill.failed{background:rgba(239,68,68,.15);color:var(--red);}
.pill.src{background:rgba(124,58,237,.15);color:#b794f6;}
.est{color:var(--amber);font-size:0.6rem;margin-left:4px;}
.tick{color:var(--green);font-weight:700;}
.dash{color:var(--muted);}
.empty{text-align:center;color:var(--muted);padding:40px 20px;font-size:0.82rem;}

/* ---- drawer ---- */
#backdrop{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:40;}
#drawer{position:fixed;top:0;right:0;height:100vh;width:480px;max-width:100%;z-index:50;
  background:#0c111c;border-left:1px solid var(--border);overflow-y:auto;animation:slideIn .2s ease-out;}
@keyframes slideIn{from{transform:translateX(100%);}to{transform:translateX(0);}}
.dh{position:sticky;top:0;background:rgba(12,17,28,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:14px 18px;display:flex;justify-content:space-between;align-items:flex-start;gap:10px;}
.dh .who{font-weight:700;font-size:0.9rem;}
.dh .meta{font-size:0.7rem;color:var(--muted);margin-top:3px;}
.dbody{padding:16px 18px;}
.x{cursor:pointer;color:var(--muted);font-size:1.3rem;line-height:1;background:none;border:none;}
.cost-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px;}
.cost-card{border:1px solid var(--border);border-radius:12px;padding:12px;}
.cost-card.gem{border-color:rgba(124,58,237,.3);}
.cost-card.tw{border-color:rgba(0,212,255,.3);}
.cost-card .ct{font-size:0.66rem;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:8px;}
.cost-card .big{font-size:1.25rem;font-weight:800;font-family:var(--mono);}
.cost-card.gem .big{color:var(--purple);}
.cost-card.tw .big{color:var(--cyan);}
.kv{display:flex;justify-content:space-between;font-size:0.7rem;color:var(--secondary);padding:3px 0;}
.kv span:last-child{font-family:var(--mono);color:var(--text);}
.sub-h{font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin:18px 0 10px;}
.msg{padding:9px 13px;border-radius:12px;max-width:88%;font-size:0.82rem;line-height:1.5;margin-bottom:7px;animation:fadeIn .2s;}
.msg .t{display:block;font-size:0.58rem;opacity:.5;font-family:var(--mono);margin-top:3px;}
.msg.user{margin-left:auto;background:linear-gradient(135deg,rgba(0,212,255,.2),rgba(0,212,255,.08));border:1px solid rgba(0,212,255,.15);}
.msg.gemini{background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(124,58,237,.08));border:1px solid rgba(124,58,237,.15);}
.tool{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:8px;padding:9px 12px;font-size:0.72rem;color:var(--green);margin-bottom:7px;}
.tool b{font-weight:700;}
.tool pre{margin-top:5px;color:var(--secondary);font-size:0.66rem;white-space:pre-wrap;word-break:break-word;}

/* ---- toasts + skeleton ---- */
#toasts{position:fixed;top:14px;right:14px;z-index:80;display:flex;flex-direction:column;gap:8px;}
.toast{background:var(--card);backdrop-filter:blur(10px);border:1px solid var(--border);border-left-width:3px;border-radius:10px;padding:10px 14px;font-size:0.78rem;animation:fadeIn .2s;min-width:200px;}
.toast.error{border-left-color:var(--red);}
.toast.success{border-left-color:var(--green);}
.toast.info{border-left-color:var(--cyan);}
.skel{background:linear-gradient(90deg,rgba(255,255,255,.03),rgba(255,255,255,.08),rgba(255,255,255,.03));background-size:800px 100%;animation:shimmer 1.4s infinite;border-radius:8px;height:14px;}

@media(max-width:820px){.row2{grid-template-columns:1fr;}#drawer{width:100%;}}
</style>
</head>
<body>

<!-- LOGIN -->
<div id="login" class="login-wrap">
  <div class="login-card">
    <h1>Admin · Call Analytics</h1>
    <p>Enter the admin key to view call logs, transcripts and costing.</p>
    <input id="keyInput" type="password" placeholder="Admin key" autocomplete="current-password"/>
    <button class="btn" id="loginBtn">Sign in</button>
    <div class="err" id="loginErr"></div>
  </div>
</div>

<!-- DASHBOARD -->
<div id="dash" class="hidden">
  <div class="top-bar">
    <span class="brand">Admin · Call Analytics</span>
    <div class="top-actions">
      <span class="status"><span class="dot"></span><span id="updated">—</span></span>
      <button class="btn ghost" id="refreshBtn">Refresh costs</button>
      <button class="btn ghost" id="logoutBtn">Log out</button>
    </div>
  </div>

  <div class="container">
    <div class="section-title">Project costing</div>
    <div class="stats" id="stats"></div>

    <div class="section-title">Spend trend</div>
    <div class="row2">
      <div class="panel" id="trend"><h3>Cost by day (Gemini + Twilio) &amp; call volume</h3><div id="trendBody"></div></div>
      <div class="panel"><h3>Breakdown</h3><div id="breakdown"></div></div>
    </div>

    <div class="section-title">Call logs</div>
    <div class="filters">
      <input type="date" id="fromDate" title="From date"/>
      <input type="date" id="toDate" title="To date"/>
      <select id="sourceFilter"><option value="">All sources</option><option value="plivo">Phone (Plivo)</option><option value="browser">Browser</option></select>
      <input class="grow" id="search" placeholder="Search caller / call SID…"/>
      <button class="btn ghost" id="csvBtn">Export CSV</button>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th data-k="started_at">Time</th>
          <th data-k="caller">Caller</th>
          <th data-k="source">Source</th>
          <th data-k="duration_seconds" class="num">Duration</th>
          <th data-k="language">Lang</th>
          <th data-k="status">Status</th>
          <th data-k="booking_created">Coming</th>
          <th data-k="rsvp_outcome_status">RSVP</th>
          <th data-k="gemini_cost_usd" class="num">Gemini $</th>
          <th data-k="twilio" class="num">Twilio $</th>
          <th data-k="total_cost_usd" class="num">Total $</th>
        </tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div id="count" class="section-title" style="margin-top:10px;"></div>

    <div class="section-title">Callbacks
      <button class="btn ghost" id="schedToggleBtn" style="margin-left:10px;font-size:0.66rem;">Scheduler</button>
      <span id="schedPaused" style="margin-left:8px;color:#f59e0b;font-size:0.7rem;"></span>
    </div>
    <div class="table-scroll">
      <table>
        <thead><tr>
          <th>Member</th><th>Requested</th><th>Due</th><th>Status</th>
          <th class="num">Attempts</th><th>Last error</th><th>Actions</th>
        </tr></thead>
        <tbody id="cbRows"></tbody>
      </table>
    </div>
  </div>
</div>

<div id="toasts"></div>

<script>
const $=(id)=>document.getElementById(id);
const KEY=()=>localStorage.getItem('admin_key')||'';
const state={summary:null,calls:[],sort:{k:'started_at',dir:'desc'}};

/* ---------- fetch helper ---------- */
async function api(path,opts={}){
  const res=await fetch(path,{...opts,headers:{...(opts.headers||{}),'X-Admin-Key':KEY()}});
  if(res.status===401){logout();throw new Error('unauthorized');}
  if(!res.ok)throw new Error('HTTP '+res.status);
  return res;
}

/* ---------- auth ---------- */
function showDash(on){$('login').classList.toggle('hidden',on);$('dash').classList.toggle('hidden',!on);}
async function login(){
  const k=$('keyInput').value.trim();
  if(!k){$('loginErr').textContent='Enter a key';return;}
  localStorage.setItem('admin_key',k);
  try{await loadAll();showDash(true);$('loginErr').textContent='';}
  catch(e){localStorage.removeItem('admin_key');$('loginErr').textContent='Invalid key';}
}
function logout(){localStorage.removeItem('admin_key');showDash(false);}

/* ---------- formatting ---------- */
const fmtUSD=(n)=>(n==null||isNaN(n))?'—':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:4}).format(n);
const fmtUSD2=(n)=>(n==null||isNaN(n))?'—':new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(n);
const fmtNum=(n)=>(n==null||isNaN(n))?'—':new Intl.NumberFormat('en-US').format(n);
const fmtPct=(x)=>(x==null||isNaN(x))?'—':(x*100).toFixed(1)+'%';
function fmtDur(s){s=s||0;const m=Math.floor(s/60),r=Math.round(s%60);return m+':'+String(r).padStart(2,'0');}
function fmtDT(iso){if(!iso)return '—';const d=new Date(iso);return d.toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});}
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function rsvpPill(s){if(!s)return '<span class="dash">—</span>';const col={yes:'#10b981',no:'#ef4444',callback:'#f59e0b',do_not_contact:'#94a3b8'}[s]||'#94a3b8';return '<span class="pill" style="background:'+col+'22;color:'+col+'">'+esc(s)+'</span>';}
function cbPill(s){const col={pending:'#f59e0b',in_flight:'#00d4ff',completed:'#10b981',failed:'#ef4444',cancelled:'#94a3b8'}[s]||'#94a3b8';return '<span class="pill" style="background:'+col+'22;color:'+col+'">'+esc(s||'')+'</span>';}

/* ---------- callbacks ---------- */
async function loadCallbacks(){
  try{
    const r=await api('/api/admin/callbacks').then(r=>r.json());
    renderCallbacks(r);
  }catch(e){if(e.message!=='unauthorized')console.error(e);}
}
function renderCallbacks(r){
  const items=r.items||[];
  const btn=$('schedToggleBtn');
  if(btn){btn.textContent=r.scheduler_enabled?'Scheduler: ON':'Scheduler: OFF';btn.dataset.on=r.scheduler_enabled?'1':'0';}
  const pe=$('schedPaused');if(pe)pe.textContent=r.paused_until?('paused until '+fmtDT(r.paused_until)):'';
  const tb=$('cbRows');
  if(!items.length){tb.innerHTML='<tr><td colspan="7"><div class="empty">No callbacks</div></td></tr>';return;}
  tb.innerHTML=items.map(c=>{
    const cb=c.callback||{};
    let act='—';
    if(cb.status==='pending'||cb.status==='failed')act='<button class="btn ghost" onclick="cbCallNow(event,\\''+c.id+'\\')">Call now</button> <button class="btn ghost" onclick="cbCancel(event,\\''+c.id+'\\')">Cancel</button>';
    else if(cb.status==='in_flight')act='<button class="btn ghost" onclick="cbCancel(event,\\''+c.id+'\\')">Cancel</button>';
    return '<tr>'+
      '<td>'+esc(cb.to||c.caller||'—')+'</td>'+
      '<td>'+esc(cb.source_text||'')+'</td>'+
      '<td>'+fmtDT(cb.due_at)+'</td>'+
      '<td>'+cbPill(cb.status)+'</td>'+
      '<td class="num">'+(cb.attempts||0)+'/'+(cb.max_attempts||3)+'</td>'+
      '<td>'+esc(cb.last_error||'')+'</td>'+
      '<td>'+act+'</td>'+
    '</tr>';
  }).join('');
}
async function cbCancel(e,id){e.stopPropagation();try{await api('/api/admin/callbacks/'+id+'/cancel',{method:'POST'});toast('Callback cancelled','success');loadCallbacks();}catch(err){if(err.message!=='unauthorized')toast('Cancel failed','error');}}
async function cbCallNow(e,id){e.stopPropagation();try{await api('/api/admin/callbacks/'+id+'/call-now',{method:'POST'});toast('Queued for callback','success');loadCallbacks();}catch(err){if(err.message!=='unauthorized')toast('Failed','error');}}
async function schedToggle(){const on=$('schedToggleBtn').dataset.on==='1';try{const r=await api('/api/admin/scheduler/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled:!on})}).then(r=>r.json());toast('Scheduler '+(r.enabled?'enabled':'disabled'),'info');loadCallbacks();}catch(err){if(err.message!=='unauthorized')toast('Toggle failed','error');}}

/* ---------- loaders ---------- */
async function loadAll(){
  const [sum,calls]=await Promise.all([
    api('/api/admin/summary').then(r=>r.json()),
    api('/api/admin/calls'+filterQS()).then(r=>r.json()),
  ]);
  state.summary=sum;state.calls=calls.items||[];
  renderStats();renderTrend();renderBreakdown();renderRows();loadCallbacks();
  $('updated').textContent='Updated '+new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function filterQS(){
  const p=new URLSearchParams();
  if($('fromDate').value)p.set('from',$('fromDate').value);
  if($('toDate').value)p.set('to',$('toDate').value);
  if($('sourceFilter').value)p.set('source',$('sourceFilter').value);
  if($('search').value.trim())p.set('q',$('search').value.trim());
  const s=p.toString();return s?('?'+s):'';
}
async function fetchCalls(){
  try{const r=await api('/api/admin/calls'+filterQS()).then(r=>r.json());state.calls=r.items||[];renderRows();}
  catch(e){if(e.message!=='unauthorized')toast('Failed to load calls','error');}
}

/* ---------- render: stats ---------- */
function renderStats(){
  const s=state.summary||{};
  const cards=[
    {label:'Total calls',value:fmtNum(s.total_calls),sub:(s.by_source?Object.entries(s.by_source).map(([k,v])=>k+': '+v).join(' · '):'')},
    {label:'Total minutes',value:(s.total_minutes!=null?s.total_minutes.toFixed(1):'—')},
    {label:'AI (Gemini) cost',value:fmtUSD(s.gemini_cost_usd),cls:'hi',sub:'real token usage'},
    {label:'Twilio cost',value:fmtUSD(s.twilio_cost_usd),cls:'cy'},
    {label:'Total real cost',value:fmtUSD(s.total_cost_usd),cls:'cy'},
    {label:'Avg cost / call',value:fmtUSD(s.avg_cost_per_call)},
    {label:'This month',value:fmtUSD2((s.this_month||{}).cost_usd),sub:'proj '+fmtUSD2(s.projected_month_cost)},
    {label:'RSVP yes-rate',value:fmtPct(s.booking_conversion_rate),cls:'gr',sub:(s.bookings||0)+' coming'},
  ];
  $('stats').innerHTML=cards.map(c=>
    '<div class="stat '+(c.cls||'')+'"><div class="label">'+c.label+'</div><div class="value">'+c.value+'</div>'+
    (c.sub?'<div class="sub">'+esc(c.sub)+'</div>':'')+'</div>').join('');
}

/* ---------- render: trend (inline SVG) ---------- */
function renderTrend(){
  const days=(state.summary&&state.summary.by_day)||[];
  const box=$('trendBody');
  if(!days.length){box.innerHTML='<div class="empty">No spend data yet</div>';return;}
  const W=640,H=220,pad={l:46,r:38,t:14,b:26};
  const iw=W-pad.l-pad.r,ih=H-pad.t-pad.b;
  const maxCost=Math.max(...days.map(d=>d.cost_usd),0.0001);
  const maxCalls=Math.max(...days.map(d=>d.calls),1);
  const n=days.length,bw=Math.max(4,Math.min(46,iw/n*0.6));
  const x=(i)=>pad.l+(iw/n)*(i+0.5);
  const yC=(v)=>pad.t+ih-(v/maxCost)*ih;
  const yN=(v)=>pad.t+ih-(v/maxCalls)*ih;
  let bars='',line='',dots='',xlabels='';
  const step=Math.ceil(n/8);
  days.forEach((d,i)=>{
    const h=(d.cost_usd/maxCost)*ih;
    bars+='<rect x="'+(x(i)-bw/2)+'" y="'+(pad.t+ih-h)+'" width="'+bw+'" height="'+h+'" rx="3" fill="url(#g)"><title>'+d.date+' · '+fmtUSD(d.cost_usd)+' · '+d.calls+' calls</title></rect>';
    line+=(i?' L':'M')+x(i)+' '+yN(d.calls);
    dots+='<circle cx="'+x(i)+'" cy="'+yN(d.calls)+'" r="3" fill="#7c3aed"/>';
    if(i%step===0)xlabels+='<text x="'+x(i)+'" y="'+(H-8)+'" fill="#64748b" font-size="9" text-anchor="middle">'+d.date.slice(5)+'</text>';
  });
  // y axis (cost) ticks
  let yticks='';
  for(let t=0;t<=2;t++){const v=maxCost*t/2,yy=yC(v);
    yticks+='<line x1="'+pad.l+'" y1="'+yy+'" x2="'+(W-pad.r)+'" y2="'+yy+'" stroke="rgba(255,255,255,0.05)"/>'+
            '<text x="'+(pad.l-6)+'" y="'+(yy+3)+'" fill="#64748b" font-size="9" text-anchor="end">'+fmtUSD2(v)+'</text>';}
  box.innerHTML='<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none">'+
    '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#00d4ff" stop-opacity="0.9"/><stop offset="1" stop-color="#00d4ff" stop-opacity="0.25"/></linearGradient></defs>'+
    yticks+bars+'<path d="'+line+'" fill="none" stroke="#7c3aed" stroke-width="2"/>'+dots+xlabels+'</svg>'+
    '<div style="display:flex;gap:16px;font-size:0.66rem;color:#94a3b8;margin-top:6px;"><span style="color:#00d4ff;">■ cost / day</span><span style="color:#7c3aed;">● calls / day</span></div>';
}

/* ---------- render: breakdown ---------- */
function renderBreakdown(){
  const s=state.summary||{};
  const g=s.gemini_cost_usd||0,t=s.twilio_cost_usd||0,tot=(g+t)||1;
  const langs=s.by_language||{};
  let html='';
  html+=brkRow('Gemini (AI)',fmtUSD(g),g/tot,'#7c3aed');
  html+=brkRow('Twilio (telephony)',fmtUSD(t),t/tot,'#00d4ff');
  html+='<div class="sub-h" style="margin:14px 0 6px;">By language</div>';
  const le=Object.entries(langs).sort((a,b)=>b[1]-a[1]);
  if(!le.length)html+='<div class="kv"><span>—</span><span></span></div>';
  le.forEach(([k,v])=>html+='<div class="kv"><span>'+esc(k)+'</span><span>'+v+'</span></div>');
  if(s.pending_twilio_price)html+='<div class="kv" style="margin-top:10px;color:#f59e0b;"><span>Pending Twilio price</span><span>'+s.pending_twilio_price+'</span></div>';
  $('breakdown').innerHTML=html;
}
function brkRow(label,val,frac,color){
  return '<div class="brk-row"><span>'+label+'</span><span style="font-family:var(--mono)">'+val+'</span></div>'+
    '<div class="brk-bar" style="width:'+Math.max(2,frac*100).toFixed(1)+'%;background:'+color+'"></div>';
}

/* ---------- render: table ---------- */
function renderRows(){
  const tb=$('rows');
  let rows=[...state.calls];
  const {k,dir}=state.sort,mul=dir==='asc'?1:-1;
  rows.sort((a,b)=>{
    let av=a[k],bv=b[k];
    if(k==='twilio'){av=(a.twilio||{}).price_usd;bv=(b.twilio||{}).price_usd;}
    if(av==null)av=-Infinity;if(bv==null)bv=-Infinity;
    if(typeof av==='string')return av.localeCompare(bv)*mul;
    return (av-bv)*mul;
  });
  $('count').textContent=rows.length+' call'+(rows.length===1?'':'s');
  if(!rows.length){tb.innerHTML='<tr><td colspan="11"><div class="empty">No calls match your filters</div></td></tr>';return;}
  tb.innerHTML=rows.map(c=>{
    const tw=(c.twilio||{}).price_usd;
    const est=c.cost_estimated?'<span class="est" title="estimated">~</span>':'';
    return '<tr onclick="openDrawer(\\''+c.id+'\\')">'+
      '<td>'+fmtDT(c.started_at)+'</td>'+
      '<td>'+esc(c.caller||(c.source==='browser'?'Web visitor':'—'))+'</td>'+
      '<td><span class="pill src">'+esc(c.source||'')+'</span></td>'+
      '<td class="num">'+fmtDur(c.duration_seconds)+'</td>'+
      '<td>'+esc(c.language||'—')+'</td>'+
      '<td><span class="pill '+esc(c.status||'')+'">'+esc(c.status||'')+'</span></td>'+
      '<td>'+(c.booking_created?'<span class="tick">✓</span>':'<span class="dash">—</span>')+'</td>'+
      '<td>'+rsvpPill(c.rsvp_outcome_status)+'</td>'+
      '<td class="num">'+fmtUSD(c.gemini_cost_usd)+'</td>'+
      '<td class="num">'+(tw==null?'<span class="dash">—</span>':fmtUSD(tw))+'</td>'+
      '<td class="num">'+fmtUSD(c.total_cost_usd)+est+'</td>'+
    '</tr>';
  }).join('');
}
function sortBy(k){
  if(state.sort.k===k)state.sort.dir=state.sort.dir==='asc'?'desc':'asc';
  else state.sort={k,dir:'desc'};
  renderRows();
}

/* ---------- drawer ---------- */
async function openDrawer(id){
  document.body.insertAdjacentHTML('beforeend','<div id="backdrop" onclick="closeDrawer()"></div><div id="drawer"><div class="dbody"><div class="skel" style="height:120px;margin-bottom:12px;"></div><div class="skel" style="height:240px;"></div></div></div>');
  try{
    const c=await api('/api/admin/calls/'+id).then(r=>r.json());
    renderDrawer(c);
  }catch(e){closeDrawer();if(e.message!=='unauthorized')toast('Failed to load call','error');}
}
function closeDrawer(){const d=$('drawer'),b=$('backdrop');if(d)d.remove();if(b)b.remove();}
function renderDrawer(c){
  const cb=c.cost_breakdown||{},gem=cb.gemini||{},tk=gem.tokens||{},tw=cb.twilio||{};
  const tcalls=(c.tool_calls||[]).map(t=>
    '<div class="tool"><b>'+esc(t.name)+'</b><pre>'+esc(JSON.stringify(t.result,null,2)||'').slice(0,600)+'</pre></div>').join('')||'<div class="kv"><span>No tool calls</span><span></span></div>';
  const tr=(c.transcript||[]).map(m=>
    '<div class="msg '+(m.role==='user'?'user':'gemini')+'"><span>'+esc(m.text)+'</span><span class="t">'+esc((m.role||'').toUpperCase())+'</span></div>').join('')||'<div class="empty">No transcript captured</div>';
  $('drawer').innerHTML=
    '<div class="dh"><div><div class="who">'+esc(c.caller||(c.source==='browser'?'Web visitor':c.call_sid||'Call'))+'</div>'+
      '<div class="meta">'+esc(c.source)+' · '+fmtDT(c.started_at)+' · '+fmtDur(c.duration_seconds)+' · '+esc(c.status||'')+'</div></div>'+
      '<button class="x" onclick="closeDrawer()">×</button></div>'+
    '<div class="dbody">'+
      '<div class="cost-grid">'+
        '<div class="cost-card gem"><div class="ct">Gemini (AI) cost</div><div class="big">'+fmtUSD(gem.cost_usd)+'</div>'+
          '<div class="kv"><span>Audio in</span><span>'+fmtNum(tk.audio_in)+'</span></div>'+
          '<div class="kv"><span>Audio out</span><span>'+fmtNum(tk.audio_out)+'</span></div>'+
          '<div class="kv"><span>Text in</span><span>'+fmtNum(tk.text_in)+'</span></div>'+
          '<div class="kv"><span>Text out</span><span>'+fmtNum(tk.text_out)+'</span></div>'+
          '<div class="kv"><span>Thinking</span><span>'+fmtNum(tk.thoughts)+'</span></div>'+
          '<div class="kv"><span>Total tokens</span><span>'+fmtNum(tk.total)+'</span></div>'+
        '</div>'+
        '<div class="cost-card tw"><div class="ct">Twilio cost</div><div class="big">'+(tw.price_usd==null?'—':fmtUSD(tw.price_usd))+'</div>'+
          '<div class="kv"><span>Duration</span><span>'+fmtDur(tw.duration_seconds)+'</span></div>'+
          '<div class="kv"><span>Unit</span><span>'+esc(tw.price_unit||'—')+'</span></div>'+
          '<div class="kv"><span>Estimated</span><span>'+(cb.cost_estimated?'yes':'no')+'</span></div>'+
          (c.source==='twilio'?'<button class="btn ghost" style="margin-top:10px;width:100%;" onclick="refreshOne(\\''+c.id+'\\')">Refresh price</button>':'')+
        '</div>'+
      '</div>'+
      '<div class="kv" style="padding:8px 2px;"><span>Total real cost</span><span style="font-weight:700;">'+fmtUSD(cb.total_cost_usd)+'</span></div>'+
      '<div class="sub-h">Tool calls</div>'+tcalls+
      '<div class="sub-h">Transcript</div>'+tr+
      '<button class="btn ghost" style="margin-top:16px;width:100%;" onclick="exportCall(\\''+c.id+'\\')">Export call JSON</button>'+
    '</div>';
}
async function refreshOne(id){
  toast('Refreshing price…','info');
  try{const r=await api('/api/admin/calls/'+id+'/refresh',{method:'POST'}).then(r=>r.json());
    toast(r.updated?'Price updated':'Price not available yet',r.updated?'success':'info');
    await loadAll();if($('drawer'))openDrawer(id);
  }catch(e){if(e.message!=='unauthorized')toast('Refresh failed','error');}
}
async function exportCall(id){
  try{const blob=await api('/api/admin/calls/'+id+'/export').then(r=>r.blob());
    dl(blob,'call_'+id+'.json');
  }catch(e){if(e.message!=='unauthorized')toast('Export failed','error');}
}

/* ---------- exports ---------- */
function dl(blob,name){const u=URL.createObjectURL(blob);const a=document.createElement('a');a.href=u;a.download=name;a.click();URL.revokeObjectURL(u);}
function exportCSV(){
  const rows=state.calls;
  const cols=['started_at','call_sid','source','caller','duration_seconds','language','status','booking_created','gemini_cost_usd','twilio_price_usd','total_cost_usd'];
  const lines=[cols.join(',')];
  rows.forEach(c=>{
    const v=[c.started_at,c.call_sid,c.source,c.caller,c.duration_seconds,c.language,c.status,c.booking_created,c.gemini_cost_usd,(c.twilio||{}).price_usd,c.total_cost_usd];
    lines.push(v.map(x=>{x=x==null?'':String(x);return /[",\\n]/.test(x)?'"'+x.replace(/"/g,'""')+'"':x;}).join(','));
  });
  dl(new Blob([lines.join('\\n')],{type:'text/csv'}),'call_logs.csv');
}
async function refreshCosts(){
  toast('Refreshing Twilio prices…','info');
  try{const r=await api('/api/admin/refresh-costs',{method:'POST'}).then(r=>r.json());
    toast('Updated '+r.updated+' of '+r.checked+' pending','success');await loadAll();
  }catch(e){if(e.message!=='unauthorized')toast('Refresh failed','error');}
}

/* ---------- toast ---------- */
function toast(msg,type){const el=document.createElement('div');el.className='toast '+(type||'info');el.textContent=msg;$('toasts').appendChild(el);setTimeout(()=>el.remove(),4000);}

/* ---------- wire up ---------- */
$('loginBtn').onclick=login;
$('keyInput').addEventListener('keypress',e=>{if(e.key==='Enter')login();});
$('logoutBtn').onclick=logout;
$('refreshBtn').onclick=refreshCosts;
$('csvBtn').onclick=exportCSV;
$('schedToggleBtn').onclick=schedToggle;
$('sourceFilter').onchange=fetchCalls;
$('fromDate').onchange=fetchCalls;
$('toDate').onchange=fetchCalls;
let st;$('search').addEventListener('input',()=>{clearTimeout(st);st=setTimeout(fetchCalls,300);});
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>sortBy(th.dataset.k));
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawer();});

/* ---------- boot ---------- */
(async()=>{
  if(KEY()){try{await loadAll();showDash(true);}catch(e){showDash(false);}}
  else showDash(false);
})();
</script>
</body>
</html>"""


# ============ ADMIN DASHBOARD (call logs, transcripts, costing) ============

def require_admin(request: Request):
    """Gate admin endpoints with ANALYTICS_SECRET (header X-Admin-Key or ?key=)."""
    key = request.headers.get("X-Admin-Key") or request.query_params.get("key") or ""
    if not (ANALYTICS_SECRET and secrets.compare_digest(str(key), str(ANALYTICS_SECRET))):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _filters_from_request(request: Request):
    qp = request.query_params
    return {
        "source": qp.get("source") or None,
        "from": qp.get("from") or None,
        "to": qp.get("to") or None,
        "q": qp.get("q") or None,
        "booking": qp.get("booking"),
        "limit": qp.get("limit"),
        "offset": qp.get("offset"),
    }


async def _refresh_call_price(call_id):
    """Re-fetch and persist Twilio's real billed price for one call."""
    call = await store.load_call(call_id)
    if not call or call.get("source") != "twilio":
        return False
    sid = call.get("call_sid")
    if not sid or str(sid).startswith("web-"):
        return False
    loop = asyncio.get_running_loop()
    info = await loop.run_in_executor(None, pricing.fetch_twilio_price, sid)
    if not info:
        return False
    call["twilio"].update({
        "price_unit": info.get("price_unit"),
        "status": info.get("status"),
        "duration_seconds": info.get("duration_seconds"),
    })
    updated = False
    if info.get("price_usd") is not None:
        call["twilio"]["price_usd"] = info["price_usd"]
        if info.get("duration_seconds"):
            call["duration_seconds"] = info["duration_seconds"]
        total, estimated = pricing.compute_total(call)
        call["total_cost_usd"] = total
        call["cost_estimated"] = estimated
        updated = True
    await store.save_call(call)
    return updated


@app.get("/admin")
async def admin_dashboard():
    """Admin dashboard — call logs, transcripts and real costing."""
    return HTMLResponse(ADMIN_DASHBOARD_HTML)


@app.get("/api/admin/summary")
async def admin_summary(request: Request):
    require_admin(request)
    filters = _filters_from_request(request)
    return JSONResponse(await store.summary(filters))


@app.get("/api/admin/calls")
async def admin_calls(request: Request):
    require_admin(request)
    filters = _filters_from_request(request)
    if filters.get("limit") is None:
        filters["limit"] = 500
    return JSONResponse(await store.list_calls(filters))


@app.get("/api/admin/calls.csv")
async def admin_calls_csv(request: Request):
    require_admin(request)
    filters = _filters_from_request(request)
    filters["limit"] = None
    data = await store.list_calls(filters)
    buf = io.StringIO()
    cols = ["started_at", "call_sid", "source", "caller", "duration_seconds",
            "language", "status", "booking_created", "gemini_cost_usd",
            "twilio_price_usd", "total_cost_usd", "cost_estimated"]
    writer = csv.writer(buf)
    writer.writerow(cols)
    for c in data["items"]:
        writer.writerow([
            c.get("started_at"), c.get("call_sid"), c.get("source"), c.get("caller"),
            c.get("duration_seconds"), c.get("language"), c.get("status"),
            c.get("booking_created"), c.get("gemini_cost_usd"),
            (c.get("twilio") or {}).get("price_usd"), c.get("total_cost_usd"),
            c.get("cost_estimated"),
        ])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=call_logs.csv"})


@app.get("/api/admin/calls/{call_id}")
async def admin_call_detail(call_id: str, request: Request):
    require_admin(request)
    call = await store.load_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    tw = call.get("twilio") or {}
    call["cost_breakdown"] = {
        "gemini": pricing.gemini_cost_breakdown(call.get("tokens")),
        "twilio": {
            "duration_seconds": tw.get("duration_seconds") or call.get("duration_seconds"),
            "price_usd": tw.get("price_usd"),
            "price_unit": tw.get("price_unit"),
        },
        "total_cost_usd": call.get("total_cost_usd"),
        "cost_estimated": call.get("cost_estimated"),
    }
    return JSONResponse(call)


@app.get("/api/admin/calls/{call_id}/export")
async def admin_call_export(call_id: str, request: Request):
    require_admin(request)
    call = await store.load_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return Response(
        content=json.dumps(call, ensure_ascii=False, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=call_{call_id}.json"},
    )


@app.post("/api/admin/refresh-costs")
async def admin_refresh_costs(request: Request):
    require_admin(request)
    data = await store.list_calls({"source": "twilio", "limit": None})
    pending = [c for c in data["items"] if (c.get("twilio") or {}).get("price_usd") is None]
    updated = 0
    for c in pending:
        try:
            if await _refresh_call_price(c["id"]):
                updated += 1
        except Exception as e:
            logger.warning(f"refresh-costs failed for {c.get('id')}: {e}")
    return {"checked": len(pending), "updated": updated}


@app.post("/api/admin/calls/{call_id}/refresh")
async def admin_call_refresh(call_id: str, request: Request):
    require_admin(request)
    updated = await _refresh_call_price(call_id)
    return {"updated": bool(updated)}


# ============ CALLBACK MANAGEMENT ============

@app.get("/api/admin/callbacks")
async def admin_callbacks(request: Request):
    require_admin(request)
    qp = request.query_params
    statuses = None
    if qp.get("status"):
        statuses = set(s.strip() for s in qp["status"].split(",") if s.strip())
    items = await store.list_callbacks(statuses)
    state = await store.load_scheduler_state()
    return JSONResponse({
        "items": items,
        "scheduler_enabled": scheduler.is_enabled(),
        "paused_until": state.get("paused_until"),
    })


@app.post("/api/admin/callbacks/{call_id}/cancel")
async def admin_callback_cancel(call_id: str, request: Request):
    require_admin(request)
    call = await store.load_call(call_id)
    if not call or not call.get("callback"):
        raise HTTPException(status_code=404, detail="Callback not found")
    call["callback"]["status"] = "cancelled"
    await store.save_call(call)
    return {"ok": True}


@app.post("/api/admin/callbacks/{call_id}/call-now")
async def admin_callback_call_now(call_id: str, request: Request):
    require_admin(request)
    call = await store.load_call(call_id)
    if not call or not call.get("callback"):
        raise HTTPException(status_code=404, detail="Callback not found")
    cb = call["callback"]
    if cb.get("status") in ("in_flight", "completed"):
        return {"ok": False, "error": f"callback is {cb.get('status')}"}
    cb["status"] = "pending"
    cb["due_at"] = datetime.now(timezone.utc).isoformat()
    cb["next_retry_at"] = None
    cb["attempts"] = 0          # operator override → give it fresh attempts
    cb["last_error"] = None
    await store.save_call(call)
    return {"ok": True}


@app.post("/api/admin/scheduler/toggle")
async def admin_scheduler_toggle(request: Request):
    require_admin(request)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    enabled = bool(body.get("enabled", not scheduler.is_enabled()))
    scheduler.set_override(enabled)
    return {"ok": True, "enabled": enabled}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

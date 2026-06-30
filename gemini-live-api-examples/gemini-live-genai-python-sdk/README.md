# EO Gujarat — "An Evening with Radha" (Gemini Live voice demo)

A real-time **voice demo** built on the Gemini Live API ([Google Gen AI Python SDK](https://github.com/googleapis/python-genai)
backend + vanilla-JS frontend). **Radha**, an AI host with a warm female **Indian-English** voice,
calls a guest and personally invites them to the **EO Gujarat evening in Ahmedabad on the 10th of
July**. The guest answers **"Yes"** or **"No"** out loud; Radha responds and her RSVP is captured
live on screen. Calls can run **in the browser** or be placed to a **real phone via Twilio**.

The FastAPI backend proxies the browser/phone WebSocket to Gemini, records each call, and tracks
token + Twilio cost on an admin dashboard.

## Quick Start

```bash
# 1. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 2. Set your Gemini API key (a .env file is easiest — see .env.example)
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 3. Start the server
uv run main.py

# 4. Open the demo and click "Start the call" (allow the microphone)
#    http://localhost:8000
```

## How the demo works

1. **Invite screen** — a branded EO Gujarat card. Click **Start the call** (browser), or enter a
   number and **Get Radha to call your phone** (Twilio).
2. **Radha greets first** — on connect she opens with her invitation and asks if she'll see you on
   the 10th.
3. **You answer by voice** — Radha replies warmly for *Yes* / graciously for *No*, and silently
   calls the `record_rsvp` tool.
4. **Live RSVP card + transcript + a glowing "Radha" orb** update in real time.
5. **Call summary** — the end screen shows the decision, duration and full transcript.

## Customising

| What | Where |
| --- | --- |
| Radha's persona / script | `SYSTEM_INSTRUCTION` in `gemini_live.py` |
| Voice (default **Aoede**, female) | `voice_name=` in `gemini_live.py` (`start_session`) |
| Accent / language (default **en-IN**) | `language_code=` in `gemini_live.py` |
| Model id | `MODEL` in `.env` (default `gemini-3.1-flash-live-preview`) |
| RSVP tool | `record_rsvp` in `gemini_live.py` (declaration) + `handle_record_rsvp` in `main.py` |
| Event details (date/city/host) | `frontend/index.html` + `EVENT` in `main.py` |
| Branding / theme | `frontend/index.html`, `frontend/style.css` (gold-on-navy "evening" palette) |

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /` | The invite / call UI |
| `WS /ws` | Browser call (proxied to Gemini Live) |
| `POST /call-me` | Place an outbound Twilio call to a phone number |
| `GET/POST /twilio/voice`, `WS /twilio/media-stream` | Twilio phone-call bridge |
| `GET /live`, `WS /live/ws` | Live transcript viewer for phone calls |
| `GET /admin` (+ `/api/admin/*`) | Call logs + token/Twilio cost (key = `ANALYTICS_SECRET`) |

## Configuration
Set values via environment variables or a `.env` file (see `.env.example`). At minimum set
`GEMINI_API_KEY`. For the phone path, set the `TWILIO_*` values and a public `PUBLIC_URL` so Twilio
can reach `/twilio/voice`.

## Project structure
```
/
├── main.py            # FastAPI server: /ws, Twilio, /call-me, /live, /admin, EVENT data + record_rsvp
├── gemini_live.py     # Gemini Live wrapper: Radha persona, record_rsvp tool, voice/accent
├── recorder.py        # Per-call recording (RSVP outcome flag)
├── store.py, pricing.py, twilio_handler.py
└── frontend/
    ├── index.html     # Invite / call / summary UI
    ├── style.css      # EO Gujarat "evening" theme
    ├── main.js        # App flow: orb, RSVP, transcript, Twilio "call me"
    ├── gemini-client.js, media-handler.js, pcm-processor.js   # transport (unchanged)
```

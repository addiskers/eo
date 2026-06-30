"""
Plivo Voice <-> Gemini Live bridge.

Handles:
- Plivo bidirectional Audio Streaming WebSocket (mulaw 8kHz)
- Audio conversion: mulaw 8kHz <-> PCM 16kHz (Gemini input) / PCM 24kHz (Gemini output)
- Bridges the two in real-time, with 20ms-paced outbound frames + barge-in (clearAudio)
- Pure Python audio conversion (no audioop, works on Python 3.13+)

Plivo media-stream protocol (JSON text frames):
  inbound  start : {"event":"start","start":{"streamId","callId","mediaFormat":{...}},"extra_headers":"k=v,k=v"}
  inbound  media : {"event":"media","media":{"payload":"<base64 mulaw>","track":"inbound"}}
  inbound  stop  : {"event":"stop", ...}
  outbound play  : {"event":"playAudio","streamId":"<id>","media":{"contentType":"audio/x-mulaw","sampleRate":8000,"payload":"<b64>"}}
  outbound clear : {"event":"clearAudio","streamId":"<id>"}   <- barge-in
Modeled on gvox-voice-proxy/app/plivo_protocol.py.
"""

import asyncio
import array
import base64
import json
import logging
import struct
import time
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# ===== Mulaw codec tables (ITU-T G.711) =====

# Mulaw -> Linear PCM16 decode table (256 entries)
_MULAW_DECODE = array.array("h")  # signed short
for _i in range(256):
    _v = ~_i
    _sign = _v & 0x80
    _exponent = (_v >> 4) & 0x07
    _mantissa = _v & 0x0F
    _sample = ((_mantissa << 3) + 0x84) << _exponent
    _sample -= 0x84
    if _sign:
        _sample = -_sample
    _MULAW_DECODE.append(max(-32768, min(32767, _sample)))

# Linear PCM16 -> Mulaw encode
_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635
_MULAW_EXP_TABLE = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]


def _pcm16_to_mulaw_sample(sample: int) -> int:
    """Encode one PCM16 sample to mulaw byte."""
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP
    sample += _MULAW_BIAS
    exponent = _MULAW_EXP_TABLE[(sample >> 7) & 0xFF]
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


# ===== Audio conversion functions =====

def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz (Plivo) -> PCM 16-bit 16kHz (Gemini input)."""
    samples_8k = [_MULAW_DECODE[b] for b in mulaw_bytes]
    # Upsample 8kHz -> 16kHz by linear interpolation
    samples_16k = []
    for i in range(len(samples_8k)):
        samples_16k.append(samples_8k[i])
        if i + 1 < len(samples_8k):
            samples_16k.append((samples_8k[i] + samples_8k[i + 1]) >> 1)
        else:
            samples_16k.append(samples_8k[i])
    return struct.pack(f"<{len(samples_16k)}h", *samples_16k)


def pcm24k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert PCM 16-bit 24kHz (Gemini output) -> mulaw 8kHz (Plivo)."""
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes)
    # Downsample 24kHz -> 8kHz (take every 3rd sample)
    samples_8k = samples[::3]
    return bytes(_pcm16_to_mulaw_sample(s) for s in samples_8k)


# 20ms of mulaw @ 8kHz = 160 bytes per frame
ULAW_FRAME_BYTES = 160
ULAW_FRAME_S = 0.020


class PlivoMediaBridge:
    """Bridges a Plivo bidirectional Audio Stream WebSocket with a Gemini Live session."""

    def __init__(self, websocket, gemini_client, text_trigger, on_event=None):
        self.ws = websocket
        self.gemini = gemini_client
        self.stream_id = None
        self.call_id = ""
        self.caller = ""
        self.generation = 0
        self.text_trigger = text_trigger
        self.on_event = on_event  # async callback for live transcript

        # Queues for Gemini
        self.audio_input_queue = asyncio.Queue()
        self.video_input_queue = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()

        # Outbound (to Plivo) paced 20ms mulaw frames
        self._out_frames = asyncio.Queue()
        self._residual = bytearray()
        self._started = False

    # ---- outbound (Gemini -> Plivo) ----

    async def audio_output_callback(self, data: bytes):
        """Gemini produced audio (24k PCM16). Convert to mulaw 8k and chunk into 20ms frames."""
        if not self.stream_id:
            return
        try:
            self._residual.extend(pcm24k_to_mulaw(data))
            while len(self._residual) >= ULAW_FRAME_BYTES:
                frame = bytes(self._residual[:ULAW_FRAME_BYTES])
                del self._residual[:ULAW_FRAME_BYTES]
                await self._out_frames.put(frame)
        except Exception as e:
            logger.error(f"Error queuing audio for Plivo: {e}")

    async def _outbound_sender(self):
        """Send queued mulaw frames to Plivo, paced at 20ms for smooth playout."""
        next_t = None
        try:
            while True:
                frame = await self._out_frames.get()
                if not self.stream_id:
                    continue
                payload = base64.b64encode(frame).decode("ascii")
                await self.ws.send_json({
                    "event": "playAudio",
                    "streamId": self.stream_id,
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": payload,
                    },
                })
                now = time.monotonic()
                next_t = (next_t or now) + ULAW_FRAME_S
                delay = next_t - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.1:
                    next_t = time.monotonic()  # fell behind, resync
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Plivo outbound sender error: {e}")

    def _drain_outbound(self):
        self._residual.clear()
        try:
            while True:
                self._out_frames.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def audio_interrupt_callback(self):
        """Barge-in: drop queued agent audio + tell Plivo to flush its playout."""
        if not self.stream_id:
            return
        self._drain_outbound()
        try:
            await self.ws.send_json({"event": "clearAudio", "streamId": self.stream_id})
        except Exception:
            pass

    # ---- inbound (Plivo -> Gemini) ----

    @staticmethod
    def _extract_header(data: dict, start: dict, name: str) -> str:
        name = name.lower()
        raw = (data.get("extra_headers") or data.get("extraHeaders")
               or start.get("extra_headers") or start.get("extraHeaders"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                if str(k).strip().lower() == name:
                    return unquote(str(v))
            return ""
        if isinstance(raw, str) and raw:
            for pair in raw.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip().lower() == name:
                        return unquote(v.strip())
        return ""

    async def handle_plivo_messages(self):
        """Receive messages from the Plivo Audio Stream WebSocket."""
        try:
            while True:
                message = await self.ws.receive_text()
                data = json.loads(message)
                event = str(data.get("event") or data.get("type") or "").lower()

                if event == "start":
                    start = data.get("start") if isinstance(data.get("start"), dict) else data
                    self.stream_id = str(start.get("streamId") or start.get("stream_id")
                                         or data.get("streamId") or "")
                    self.call_id = str(start.get("callId") or start.get("call_id") or "")
                    self.caller = self._extract_header(data, start, "x-caller")
                    gen_raw = self._extract_header(data, start, "x-callback-gen")
                    try:
                        self.generation = int(gen_raw) if gen_raw else 0
                    except ValueError:
                        self.generation = 0
                    logger.info(f"Plivo stream started: stream_id={self.stream_id}, "
                                f"call={self.call_id}, caller={self.caller}, gen={self.generation}")
                    if not self._started:
                        self._started = True
                        await self._emit({"type": "call_start", "call_sid": self.call_id or "",
                                          "caller": self.caller or "",
                                          "generation": self.generation})
                    # Trigger the AI to start talking
                    await self.text_input_queue.put(self.text_trigger)

                elif event == "media":
                    media = data.get("media") or {}
                    payload = media.get("payload")
                    if payload:
                        mulaw_bytes = base64.b64decode(payload)
                        await self.audio_input_queue.put(mulaw_to_pcm16k(mulaw_bytes))

                elif event == "dtmf":
                    pass

                elif event == "stop":
                    logger.info("Plivo stream stopped")
                    break

        except Exception as e:
            logger.error(f"Plivo receive error: {e}")

    async def _emit(self, event):
        """Send event to live transcript watchers."""
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception:
                pass

    async def run(self):
        """Run the bridge: Plivo <-> Gemini."""
        plivo_task = asyncio.create_task(self.handle_plivo_messages())
        sender_task = asyncio.create_task(self._outbound_sender())

        try:
            async for event in self.gemini.start_session(
                audio_input_queue=self.audio_input_queue,
                video_input_queue=self.video_input_queue,
                text_input_queue=self.text_input_queue,
                audio_output_callback=self.audio_output_callback,
                audio_interrupt_callback=self.audio_interrupt_callback,
            ):
                if event:
                    await self._emit(event)
                    if event.get("type") == "error":
                        logger.error(f"Gemini error during Plivo call: {event}")
                        break
        except Exception as e:
            logger.error(f"Gemini session error: {e}")
        finally:
            if self._started:
                await self._emit({"type": "call_end"})
            plivo_task.cancel()
            sender_task.cancel()
            logger.info("Plivo-Gemini bridge closed")

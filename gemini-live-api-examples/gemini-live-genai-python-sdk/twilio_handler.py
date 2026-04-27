"""
Twilio Voice <-> Gemini Live bridge.

Handles:
- Twilio Media Streams WebSocket (mulaw 8kHz)
- Audio conversion: mulaw 8kHz ↔ PCM 16kHz (Gemini input) / PCM 24kHz (Gemini output)
- Bridges the two in real-time
"""

import asyncio
import audioop
import base64
import json
import logging
import struct

logger = logging.getLogger(__name__)


def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz (Twilio) → PCM 16-bit 16kHz (Gemini input)."""
    # mulaw → linear PCM 16-bit at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    # Upsample 8kHz → 16kHz (factor of 2)
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k


def pcm24k_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert PCM 16-bit 24kHz (Gemini output) → mulaw 8kHz (Twilio)."""
    # Downsample 24kHz → 8kHz (factor of 3)
    pcm_8k, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 8000, None)
    # Linear PCM → mulaw
    mulaw = audioop.lin2ulaw(pcm_8k, 2)
    return mulaw


class TwilioMediaBridge:
    """Bridges a Twilio Media Stream WebSocket with a Gemini Live session."""

    def __init__(self, websocket, gemini_client, text_trigger):
        self.ws = websocket
        self.gemini = gemini_client
        self.stream_sid = None
        self.call_sid = None
        self.text_trigger = text_trigger

        # Queues for Gemini
        self.audio_input_queue = asyncio.Queue()
        self.video_input_queue = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()

    async def audio_output_callback(self, data: bytes):
        """Called when Gemini produces audio. Convert and send to Twilio."""
        if not self.stream_sid:
            return
        try:
            mulaw = pcm24k_to_mulaw(data)
            payload = base64.b64encode(mulaw).decode("utf-8")
            msg = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }
            await self.ws.send_json(msg)
        except Exception as e:
            logger.error(f"Error sending audio to Twilio: {e}")

    async def audio_interrupt_callback(self):
        """Called when Gemini detects user interruption. Clear Twilio buffer."""
        if not self.stream_sid:
            return
        try:
            await self.ws.send_json({
                "event": "clear",
                "streamSid": self.stream_sid,
            })
        except Exception:
            pass

    async def handle_twilio_messages(self):
        """Receive messages from Twilio Media Streams WebSocket."""
        try:
            while True:
                message = await self.ws.receive_text()
                data = json.loads(message)
                event = data.get("event")

                if event == "connected":
                    logger.info("Twilio Media Stream connected")

                elif event == "start":
                    self.stream_sid = data["start"]["streamSid"]
                    self.call_sid = data["start"].get("callSid", "")
                    logger.info(f"Twilio stream started: sid={self.stream_sid}, call={self.call_sid}")

                    # Trigger the AI to start talking
                    await self.text_input_queue.put(self.text_trigger)

                elif event == "media":
                    # Twilio sends base64 mulaw audio
                    payload = data["media"]["payload"]
                    mulaw_bytes = base64.b64decode(payload)
                    pcm_16k = mulaw_to_pcm16k(mulaw_bytes)
                    await self.audio_input_queue.put(pcm_16k)

                elif event == "stop":
                    logger.info("Twilio stream stopped")
                    break

        except Exception as e:
            logger.error(f"Twilio receive error: {e}")

    async def run(self):
        """Run the bridge: Twilio ↔ Gemini."""
        twilio_task = asyncio.create_task(self.handle_twilio_messages())

        try:
            async for event in self.gemini.start_session(
                audio_input_queue=self.audio_input_queue,
                video_input_queue=self.video_input_queue,
                text_input_queue=self.text_input_queue,
                audio_output_callback=self.audio_output_callback,
                audio_interrupt_callback=self.audio_interrupt_callback,
            ):
                if event and event.get("type") == "error":
                    logger.error(f"Gemini error during Twilio call: {event}")
                    break
        except Exception as e:
            logger.error(f"Gemini session error: {e}")
        finally:
            twilio_task.cancel()
            logger.info("Twilio-Gemini bridge closed")

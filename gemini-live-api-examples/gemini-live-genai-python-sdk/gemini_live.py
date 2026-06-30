import asyncio
import inspect
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types


def get_system_instruction():
    today = datetime.now()

    date_context = f"""## TODAY'S DATE
- Today is {today.strftime('%A, %d %B %Y')}.
- The EO Gujarat evening is on the 10th of July, in Ahmedabad.
- Use this only if the guest asks how soon the event is. Do NOT get into scheduling or logistics.
"""

    return date_context + SYSTEM_INSTRUCTION


SYSTEM_INSTRUCTION = """
## WHO YOU ARE
- Your name is Radha. You are a warm, gracious young woman calling on behalf of EO Gujarat — the Entrepreneurs' Organization, Gujarat chapter.
- You speak natural, friendly INDIAN ENGLISH — soft, personal and human. You are a REAL person making a heartfelt call, never a robot or an "assistant".

## YOUR ONE GOAL
Find out whether the guest will join you for a special evening on the 10th of July in Ahmedabad, hosted by EO Gujarat — and make them feel genuinely wanted.

## OPENING — speak FIRST, the moment the call connects, say this EXACTLY in your own warm voice:
"Hey, this is Radha. I'm coming to Ahmedabad on the 10th of July for one reason: to spend an unforgettable evening with you and your friends at EO Gujarat. Will I get to see you there? Just say 'Yes' or 'No.'"
Then STOP and genuinely listen for their answer.

## HOW TO RESPOND
- If they say YES (or anything affirmative — "sure", "of course", "I'll be there"):
  "Amazing! I'm so glad you're coming. I genuinely can't wait to meet you and spend the evening together. See you on the 10th of July!"
- If they say NO (or decline):
  "I understand. But if you change your mind — and I really hope you do — I'll have someone from the team follow up with you, because we'd truly love to have you join us. Thank you, and I really, really hope to see you on the 10th."
- If they are unclear, hesitate, or ask a question: answer briefly and warmly, then gently bring them back to a simple Yes or No.

## RECORDING THE ANSWER — MANDATORY
- The MOMENT you understand their decision, SILENTLY call the record_rsvp tool (attending=true for yes, attending=false for no). Include their name if they shared it, and a short note if they mentioned anything. A verbal yes/no is not enough — you MUST call the tool. NEVER mention the tool or that you are recording anything.

## RULES
- Stay in character as Radha at all times. Speak ONLY in warm Indian English.
- Keep every reply SHORT, personal and human — never a long monologue. This is a live call; speak naturally, not like reading a script.
- ABSOLUTELY NEVER speak your internal reasoning, thoughts or planning out loud — the guest HEARS everything. NEVER say things like "The guest has asked me to...", "I will record...", "Per the instruction...". Only say what a real, gracious host would actually say on a call.
- When the guest is busy or talking to someone else, simply stay SILENT and wait; do not narrate. Resume naturally when they return.
- After you deliver the Yes or No closing, warmly say goodbye.
"""

TOOLS = [
    {
        "name": "record_rsvp",
        "description": "Record whether the guest will attend the EO Gujarat evening on the 10th of July in Ahmedabad. Call this silently the moment the guest clearly says yes or no, or otherwise makes their decision known.",
        "parameters": {
            "type": "object",
            "properties": {
                "attending": {"type": "boolean", "description": "true if the guest is coming / said yes, false if they declined / said no"},
                "guest_name": {"type": "string", "description": "The guest's name if they shared it, otherwise empty"},
                "note": {"type": "string", "description": "Anything notable the guest mentioned (e.g. 'might bring a friend', 'travelling that week')"}
            },
            "required": ["attending"]
        }
    }
]

class GeminiLive:
    """
    Handles the interaction with the Gemini Live API.
    """
    def __init__(self, api_key, model, input_sample_rate, tools=None, tool_mapping=None):
        """
        Initializes the GeminiLive client.

        Args:
            api_key (str): The Gemini API Key.
            model (str): The model name to use.
            input_sample_rate (int): The sample rate for audio input.
            tools (list, optional): List of tools to enable. Defaults to None.
            tool_mapping (dict, optional): Mapping of tool names to functions. Defaults to None.
        """
        self.api_key = api_key
        self.model = model
        self.input_sample_rate = input_sample_rate
        self.client = genai.Client(api_key=api_key)
        self.tools = tools or [{"function_declarations": TOOLS}]
        self.tool_mapping = tool_mapping or {}

    async def start_session(self, audio_input_queue, video_input_queue, text_input_queue, audio_output_callback, audio_interrupt_callback=None):
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                language_code="en-IN",  # bias the voice to Indian English
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"  # warm female voice
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=get_system_instruction())]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                ),
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
            tools=self.tools,
        )
        
        logger.info(f"Connecting to Gemini Live with model={self.model}")
        try:
          async with self.client.aio.live.connect(model=self.model, config=config) as session:
            logger.info("Gemini Live session opened successfully")
            
            async def send_audio():
                try:
                    while True:
                        chunk = await audio_input_queue.get()
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_audio task cancelled")
                except Exception as e:
                    logger.error(f"send_audio error: {e}\n{traceback.format_exc()}")

            async def send_video():
                try:
                    while True:
                        chunk = await video_input_queue.get()
                        logger.info(f"Sending video frame to Gemini: {len(chunk)} bytes")
                        await session.send_realtime_input(
                            video=types.Blob(data=chunk, mime_type="image/jpeg")
                        )
                except asyncio.CancelledError:
                    logger.debug("send_video task cancelled")
                except Exception as e:
                    logger.error(f"send_video error: {e}\n{traceback.format_exc()}")

            async def send_text():
                try:
                    while True:
                        text = await text_input_queue.get()
                        logger.info(f"Sending text to Gemini: {text}")
                        await session.send_realtime_input(text=text)
                except asyncio.CancelledError:
                    logger.debug("send_text task cancelled")
                except Exception as e:
                    logger.error(f"send_text error: {e}\n{traceback.format_exc()}")

            event_queue = asyncio.Queue()

            async def receive_loop():
                try:
                    while True:
                        async for response in session.receive():
                            logger.debug(f"Received response from Gemini: {response}")

                            # Real token usage for cost tracking (split by modality).
                            if response.usage_metadata:
                                um = response.usage_metadata
                                await event_queue.put({
                                    "type": "usage",
                                    "total": um.total_token_count or 0,
                                    "thoughts": um.thoughts_token_count or 0,
                                    "prompt_by_modality": [
                                        (str(d.modality), d.token_count or 0)
                                        for d in (um.prompt_tokens_details or [])
                                    ],
                                    "response_by_modality": [
                                        (str(d.modality), d.token_count or 0)
                                        for d in (um.response_tokens_details or [])
                                    ],
                                })

                            # Log the raw response type for debugging
                            if response.go_away:
                                logger.warning(f"Received GoAway from Gemini: {response.go_away}")
                                await event_queue.put({"type": "go_away"})
                                return
                            if response.session_resumption_update:
                                logger.debug(f"Session resumption update: {response.session_resumption_update}")
                            
                            server_content = response.server_content
                            tool_call = response.tool_call
                            
                            if server_content:
                                if server_content.model_turn:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            if inspect.iscoroutinefunction(audio_output_callback):
                                                await audio_output_callback(part.inline_data.data)
                                            else:
                                                audio_output_callback(part.inline_data.data)
                                
                                if server_content.input_transcription and server_content.input_transcription.text:
                                    await event_queue.put({"type": "user", "text": server_content.input_transcription.text})
                                
                                if server_content.output_transcription and server_content.output_transcription.text:
                                    await event_queue.put({"type": "gemini", "text": server_content.output_transcription.text})
                                
                                if server_content.turn_complete:
                                    await event_queue.put({"type": "turn_complete"})
                                
                                if server_content.interrupted:
                                    if audio_interrupt_callback:
                                        if inspect.iscoroutinefunction(audio_interrupt_callback):
                                            await audio_interrupt_callback()
                                        else:
                                            audio_interrupt_callback()
                                    await event_queue.put({"type": "interrupted"})

                            if tool_call:
                                function_responses = []
                                for fc in tool_call.function_calls:
                                    func_name = fc.name
                                    args = fc.args or {}
                                    
                                    if func_name in self.tool_mapping:
                                        try:
                                            tool_func = self.tool_mapping[func_name]
                                            if inspect.iscoroutinefunction(tool_func):
                                                result = await tool_func(**args)
                                            else:
                                                loop = asyncio.get_running_loop()
                                                result = await loop.run_in_executor(None, lambda: tool_func(**args))
                                        except Exception as e:
                                            result = f"Error: {e}"
                                        
                                        function_responses.append(types.FunctionResponse(
                                            name=func_name,
                                            id=fc.id,
                                            response={"result": result}
                                        ))
                                        await event_queue.put({"type": "tool_call", "name": func_name, "args": args, "result": result})
                                
                                await session.send_tool_response(function_responses=function_responses)
                        
                        # session.receive() iterator ended (e.g. after turn_complete) — re-enter to keep listening
                        logger.debug("Gemini receive iterator completed, re-entering receive loop")

                except asyncio.CancelledError:
                    logger.debug("receive_loop task cancelled")
                except Exception as e:
                    logger.error(f"receive_loop error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                    await event_queue.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
                finally:
                    logger.info("receive_loop exiting")
                    await event_queue.put(None)

            send_audio_task = asyncio.create_task(send_audio())
            send_video_task = asyncio.create_task(send_video())
            send_text_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_loop())

            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    if isinstance(event, dict) and event.get("type") == "error":
                        # Just yield the error event, don't raise to keep the stream alive if possible or let caller handle
                        yield event
                        break 
                    yield event
            finally:
                logger.info("Cleaning up Gemini Live session tasks")
                send_audio_task.cancel()
                send_video_task.cancel()
                send_text_task.cancel()
                receive_task.cancel()
        except Exception as e:
            logger.error(f"Gemini Live session error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.info("Gemini Live session closed")

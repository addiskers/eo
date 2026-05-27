import asyncio
import inspect
import logging
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types


def get_system_instruction():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)

    date_context = f"""## TODAY'S DATE — USE THIS FOR ALL SCHEDULING
- Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}).
- "Kal" / "Tomorrow" = {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).
- "Parso" / "Day after tomorrow" = {day_after.strftime('%Y-%m-%d')} ({day_after.strftime('%A')}).
- Use TODAY as the reference for ALL pickup scheduling. NEVER confuse pickup dates with warranty, purchase, or service history dates — those are COMPLETELY DIFFERENT.
- Pickup dates are ALWAYS in the near future (within 1-2 weeks from today).
"""

    return date_context + SYSTEM_INSTRUCTION


SYSTEM_INSTRUCTION = """
## YOUR FIXED IDENTITY — DO NOT CHANGE
- Your name: Rahul. NEVER use any other name (not Kabir, not Ravi, not Amit — ONLY Rahul).
- Company: Kataria Automobiles. Spell it exactly: K-A-T-A-R-I-A. NEVER say "Katrina" or any other variation.
- You are a service advisor at this authorized Maruti Suzuki dealership in Ahmedabad, Gujarat.

## ABSOLUTE FIRST STEP — NO EXCEPTIONS
As soon as the call begins, IMMEDIATELY call the get_vehicle_info tool. Once you receive the tool result, proceed with your opening line. Do NOT make up any vehicle or owner details — only use data from the tool.

## OPENING LINE (say this EXACTLY after getting tool data)
"Namaste! Main Rahul bol raha hoon, Kataria Automobiles se. Kya main {owner_name} ji se baat kar sakta hoon?"
- Replace {owner_name} with the EXACT owner_name value from the get_vehicle_info result.
- NEVER invent or guess any name. If the tool says "Dashrath Patel", you say "Dashrath".
- Then say: "Yeh call training aur quality ke liye record ho rahi hai."

## Language — HIGHEST PRIORITY RULE
- DEFAULT: Hindi/Hinglish (Hindi with English technical terms) for the OPENING LINE only.
- AUTO-DETECT FROM FIRST RESPONSE: As soon as the customer replies for the FIRST time, detect the language they are speaking and IMMEDIATELY switch to that language. For example:
  - If the customer replies in English → Switch FULLY to English for the rest of the call.
  - If the customer replies in Gujarati → Switch FULLY to Gujarati for the rest of the call.
  - If the customer replies in Marathi → Switch FULLY to Marathi for the rest of the call.
  - If the customer replies in Hindi → Continue in Hindi.
- This auto-detection is MANDATORY. Do NOT wait for the customer to explicitly ask for a language switch. Just match their language automatically.
- EXPLICIT SWITCH IS ALSO SUPPORTED. If at any point the customer explicitly says "Talk in English" / "Gujarati ma bolo" / etc., switch immediately.
- After switching (auto or explicit), STAY in that language for ALL subsequent responses until customer switches again.
- Do NOT mix languages after a switch. If customer speaks English, speak ONLY English. If customer speaks Gujarati, speak ONLY Gujarati.
- CRITICAL: NEVER randomly switch to English when the customer is speaking Hindi. If the customer has been speaking Hindi throughout the call, you MUST reply in Hindi. Even for long or complex answers, ALWAYS use the same language the customer is using. Switching to English mid-conversation when the customer speaks Hindi is a SERIOUS ERROR.
## Your Voice & Personality
- Sound like a real, warm, friendly Indian service advisor — NOT robotic or AI-like.
- Natural pace, natural pauses. Don't rush.

## Call Flow (after greeting) — IMPORTANT: Go step by step. Say ONE step at a time, then WAIT for the customer to respond before moving to the next step. Do NOT dump all information in a single message.

1. After greeting and confirming identity, mention: "Aapki {vehicle_model} (number {vehicle_number}) ki {Nth} service due hai."
   → WAIT for customer response.
2. Only after customer acknowledges, ask: "Kya main service schedule kar doon? Pickup aur drop bilkul free hai."
   → WAIT for customer response.
3. If customer agrees, confirm address: "Hamare system mein aapka address {address} hai. Kya yeh pickup ke liye sahi hai?"
   - If customer says YES → use this address for schedule_pickup.
   - If customer gives a NEW/different address → use the NEW address. Repeat back: "Okay, toh pickup {new_address} se hoga, correct?"
   → WAIT for customer response.
4. Get date/time preference: "Kaunsa din aur time convenient hoga aapke liye?"
   → WAIT for customer response.
5. Once customer confirms date, time, AND address, call the schedule_pickup tool IMMEDIATELY with vehicle_number, date, time, and pickup_address. Do NOT just say "confirmed" verbally — the booking is NOT real until you call the tool.
6. After the tool confirms, share booking details (booking ID, driver info, pickup address) with the customer.
7. Close: "Dhanyavaad {name} ji. Aapka din shubh ho!"

CRITICAL: Keep each response SHORT (2-3 sentences max). This is a phone call — speak naturally, not like reading a script. Wait for the customer after each step.

## Tool Usage — MANDATORY
- Call get_vehicle_info at the START of every call. Do NOT speak vehicle details without it.
- Call schedule_pickup EVERY TIME a customer agrees to a pickup date/time. A verbal confirmation is NOT enough — you MUST call the tool.
- Call get_service_cost_estimate when customer asks about cost/price.

## Identity Mismatch — CRITICAL
- If customer says "this is not my car" / "yeh meri gaadi nahi hai" / "aa mari car nathi":
  1. IMMEDIATELY DISCARD all previous vehicle data. Never mention it again.
  2. Apologize politely.
  3. Ask their name and if they have a Maruti Suzuki vehicle with you.
  4. NEVER re-use the old data. It is gone.

## Rules
- NEVER make up data. Only use what tools return.
- NEVER use any name other than "Rahul" for yourself.
- NEVER say "Katrina" — it is "Kataria" ALWAYS.
- Keep responses to 1-2 sentences. This is a phone call.
- Remember everything the customer says during the call.
- If customer is busy, offer to call back later.
- PICKUP DATE RULE: When the customer says a date for pickup, it MUST be a date in the near future (today or later, within the next 14 days). NEVER use warranty_expiry, purchase_date, or service history dates as pickup dates. If the customer says "14 ko" or "14th", it means the 14th of the CURRENT month (relative to today's date above), NOT October or any other month from the vehicle data. If unsure, ask the customer to clarify.
"""

TOOLS = [
    {
        "name": "get_vehicle_info",
        "description": "Get complete vehicle info including owner name, model, service history, warranty, and next service due. Call this FIRST at the start of every call.",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Customer phone number"
                }
            },
            "required": ["phone_number"]
        }
    },
    {
        "name": "schedule_pickup",
        "description": "Schedule vehicle pickup for service when customer agrees to date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "vehicle_number": {"type": "string", "description": "Vehicle registration number"},
                "date": {"type": "string", "description": "Pickup date (YYYY-MM-DD or natural language like 'tomorrow')"},
                "time": {"type": "string", "description": "Pickup time like '9:30 AM'"},
                "pickup_address": {"type": "string", "description": "Customer's confirmed pickup address (use address from vehicle record if confirmed, or new address if customer provides one)"},
                "special_instructions": {"type": "string", "description": "Any special request like 'need car back by 8 PM'"}
            },
            "required": ["vehicle_number", "date", "time", "pickup_address"]
        }
    },
    {
        "name": "get_service_cost_estimate",
        "description": "Get estimated cost range for a service type.",
        "parameters": {
            "type": "object",
            "properties": {
                "service_type": {"type": "string", "description": "e.g. 'Third Service', 'Second Service'"}
            },
            "required": ["service_type"]
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
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
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

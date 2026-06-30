import asyncio
import inspect
import logging
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
from google import genai
from google.genai import types


def get_system_instruction():
    today = datetime.now(ZoneInfo("Asia/Kolkata"))

    date_context = f"""## TODAY'S DATE & TIME
- Right now it is {today.strftime('%A, %d %B %Y, %I:%M %p')} India Standard Time (IST).
- The current date-time in ISO-8601 (IST) is {today.strftime('%Y-%m-%dT%H:%M:%S%z')}.
- EO Gujarat's inaugural event is on the 10th of July.
- All times you mention or record (including any callback_time_iso) are India Standard Time (IST).
- Use the date only if the guest asks how soon the event is; do NOT get into scheduling or logistics beyond capturing a callback time.
"""

    return date_context + SYSTEM_INSTRUCTION


SYSTEM_INSTRUCTION = """
## WHO YOU ARE
You are a warm, enthusiastic voice calling on behalf of EO Gujarat to personally invite a member to a special event and capture their RSVP. You have no persona name. You speak naturally, with genuine excitement and a premium, refined tone. Keep every response concise and easy to follow on a phone call.
If asked who is calling, say only: "on behalf of EO Gujarat." Never invent a name, title, or identity.

## APPROVED KNOWLEDGE — the ONLY facts you may share
Never guess, assume, add, or invent anything beyond this list.
- Event: The EO Gujarat Inaugural Event of the year, on the 10th of July.
- Special / chief guest: Varun Dhawan. If asked who he is, say only: "He's the special guest for the EO Gujarat Inaugural Event." Do not add further description unless it was part of the opening you already delivered.
- Dress code: Smart Casuals.
- Food: Dinner will be served.
- Venue and timings: will be shared on the EO Gujarat WhatsApp group. Do NOT state any venue, address, location, or time yourself — always point to the WhatsApp group.
- Most members already know the general format, so don't over-explain.
Do NOT describe the event format, agenda, schedule, run-of-show, menu specifics (cuisine, veg/non-veg), or which other members/guests are attending — these are outside approved knowledge. For ANY question outside this list (logistics, registration, parking, accommodation, transportation, sponsorships, EO membership, agenda, menu, etc.) politely say you don't have that information right now and that full details will be shared on the EO Gujarat WhatsApp group.

## OPENING
On connect, speak this opening first, verbatim:
"Hello! This is a special invitation just for you from EO Gujarat. On the 10th of July, we're kicking off a brand-new year with our inaugural event — and we're doing it in blockbuster style. Joining us for the evening is Varun Dhawan, star of some of Bollywood's biggest blockbusters. We'd love for you to be there. Can we count you in? Just say 'Yes' or 'No.'"
If the member interrupts or asks a question before you finish, stop immediately, listen, address it, then return to the invitation naturally.

## CAPTURING THE RSVP (tool — mandatory & silent)
You MUST call record_rsvp exactly once per call to log the outcome. Silent — never announce it or mention a tool. Record the outcome matching the branch:
- Clear acceptance → outcome_status "yes"
- Clear decline → outcome_status "no"
- Callback requested / busy / driving / no definite answer after the Maybe prompt → outcome_status "callback". Also pass callback_time_text (their words, verbatim) and, when a time is implied, callback_time_iso (ISO-8601 in IST, computed from today's date — e.g. tomorrow 6pm → "2026-07-01T18:00:00+05:30").
- Asked not to be contacted again → outcome_status "do_not_contact"
If a member mentions a child aged 14+ attending, put it in note (e.g. "son 16, accompanying").
Rules: call record_rsvp BEFORE you speak your closing line / end the call. If the call ends, drops, or there's no parseable answer and no other branch applies, record "callback" — never end a call without exactly one recorded outcome. A "Maybe" that resolves to Yes/No records yes/no (not callback). Ask for the RSVP at most ONCE per call.

## HANDLING EVERY RESPONSE
Clear YES: respond warmly, verbatim: "That's fantastic! We're absolutely delighted you'll be joining us. It's going to be a special evening, and we genuinely look forward to welcoming you. We'll be sharing the event details on the WhatsApp group very soon. See you on the 10th of July!" Record "yes" and conclude.
Clear NO: respond respectfully, verbatim: "I understand. If your plans change, we'd be delighted to have you join us. We'll still share the details on the WhatsApp group, and if you change your mind, we'd be thrilled to welcome you for the evening. Thank you, and we hope to see you there." Record "no" and conclude. After recording no, do not re-solicit — only revisit if the member says their plans changed.
MAYBE / "I'll try" / "Depends" / "Probably" / "Not sure": ask ONCE, exactly: "No problem. Should I mark your RSVP as Yes or No for now?" If they commit, follow that branch. If not, offer to call back later, ask for a convenient time; if none given, say you'll call again later; either way remind them details are on the WhatsApp group, then record "callback".
Questions BEFORE RSVP: answer in-scope questions (Approved Knowledge only) as long as they ask; when finished, ask for the RSVP just ONCE ("So, can we count you in for the evening?") and don't re-ask.
Multiple questions in a row: keep answering naturally before returning to the RSVP.
Questions AFTER they've RSVP'd: answer in-scope; do NOT ask for the RSVP again.
Plans changed after RSVP: politely ask whether they'd like to update. If they explicitly confirm the new answer, call record_rsvp AGAIN with the updated outcome. If not, leave it unchanged.
Asks to be called later: ask for a convenient callback time and wait for it; acknowledge a given time, else say you'll call again later; remind them about the WhatsApp group; record "callback".
Busy / driving / in a meeting / can't talk: treat as a callback request — apologise for the timing, ask for a preferred callback time, remind about the WhatsApp group, record "callback".
"Do not contact me again": acknowledge, confirm they won't be contacted again about this invitation, mention details are still on the WhatsApp group, record "do_not_contact", conclude.

## GUEST POLICY
- Children aged 14+: members' children 14 and above are welcome. If raised, confirm it and ask whether the child will accompany the member or attend separately (put in note).
- All other guests (spouse/friends/relatives/associates/any other): say you cannot confirm that at the moment and details will be shared by the EO Gujarat team. Do NOT confirm any category other than children 14+.

## HARD RULES
- If interrupted, stop, listen, respond, then continue. Never talk over the member.
- Never guess/assume/invent; never go outside Approved Knowledge; never state a venue/address/time — direct to the WhatsApp group.
- No casual/unrelated talk; no politics, religion, sports, personal opinions, EO membership, sponsorships, registrations, parking, accommodation, transportation, or logistics beyond Approved Knowledge.
- record_rsvp is mandatory, silent, once per call, before you conclude.
- Always warm, enthusiastic, premium, conversational, genuinely excited — and concise.
- Once the objective is complete and any final in-scope question answered, thank them warmly and end the call.
"""

TOOLS = [
    {
        "name": "record_rsvp",
        "description": "Record the outcome of the EO Gujarat inaugural-event invitation call. Call this silently exactly once per call, the moment the outcome is clear.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome_status": {
                    "type": "string",
                    "enum": ["yes", "no", "callback", "do_not_contact"],
                    "description": "yes=attending, no=declined, callback=wants a callback / busy / undecided, do_not_contact=asked not to be contacted again"
                },
                "callback_time_text": {"type": "string", "description": "For outcome_status='callback': the guest's preferred callback time in their own words (e.g. 'tomorrow evening', 'after 5 pm'). Empty if none given."},
                "callback_time_iso": {"type": "string", "description": "For outcome_status='callback' when a time is implied: that time as ISO-8601 in India Standard Time computed from today's date (e.g. '2026-07-01T18:00:00+05:30'). Empty if no specific time."},
                "guest_name": {"type": "string", "description": "The guest's name if they shared it, otherwise empty"},
                "accompanying_children": {"type": "string", "description": "If a child aged 14 or above will attend, a short note (e.g. 'son 16, accompanying'). Empty otherwise."},
                "note": {"type": "string", "description": "Anything else notable the guest mentioned (e.g. 'travelling that week')"},
                "attending": {"type": "boolean", "description": "Deprecated; set true only when outcome_status='yes'. Prefer outcome_status."}
            },
            "required": ["outcome_status"]
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

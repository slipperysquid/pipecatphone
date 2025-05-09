#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys

from typing import Optional

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport


from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.frames.frames import TTSSpeakFrame, EndFrame


import datetime


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

# --- Post-Call Logging State Variables ---
_call_start_time: Optional[datetime.datetime] = None
_silence_event_count: int = 0
MAX_SILENCE_PROMPTS: int = 3
_summary_logged_flag: bool = False
_active_pipeline_task: Optional[PipelineTask] = None


# --- Logging Function ---
def log_call_summary(termination_reason: str, call_id_info: str = "N/A"):
    global _call_start_time, _silence_event_count, _summary_logged_flag
    if _summary_logged_flag:
        logger.info("Call summary already logged for this session.")
        return

    logger.info("Attempting to log call summary.")
    if _call_start_time:
        call_end_time = datetime.datetime.now()
        duration = call_end_time - _call_start_time
        logger.info(f"--- Post-Call Summary (Call ID/Info: {call_id_info}) ---")
        logger.info(f"Termination Reason: {termination_reason}")
        logger.info(f"Call Start Time: {_call_start_time.isoformat()}")
        logger.info(f"Call End Time: {call_end_time.isoformat()}")
        logger.info(f"Call Duration: {str(duration)}")
        logger.info(f"Silence Events Detected (10s+): {_silence_event_count}")
        logger.info(f"----------------------------------------------------")
    else:
        logger.warning(f"Call start time not recorded for Call ID/Info: {call_id_info}, cannot log full duration summary.")
        logger.info(f"--- Post-Call Summary (Partial) (Call ID/Info: {call_id_info}) ---")
        logger.info(f"Termination Reason: {termination_reason}")
        logger.info(f"Silence Events Detected (10s+): {_silence_event_count}")
        logger.info(f"-------------------------------------------------------------")
    _summary_logged_flag = True
    
async def handle_user_idle(processor: UserIdleProcessor, retry_count: int):
    """Handle user inactivity by sending reminders and eventually ending the call."""
    global _silence_event_count, _active_pipeline_task 
    _silence_event_count += 1
    logger.info(f"User idle detected. Retry count: {retry_count}, Total silence events for this call: {_silence_event_count}")

    prompt_message = ""
    final_goodbye_message = False
    if retry_count == 1:
        prompt_message = "Are you still there?"
    elif retry_count == 2:
        prompt_message = "It seems you're not responding. Would you like to continue our conversation?"
    elif retry_count >= MAX_SILENCE_PROMPTS: 
        prompt_message = "I haven't heard from you in a while. I'll disconnect the call now. Goodbye!"
        final_goodbye_message = True

    if prompt_message:
        logger.info(f"Sending TTS prompt: '{prompt_message}'")
        try:
            await processor.push_frame(TTSSpeakFrame(prompt_message))
            logger.debug(f"Successfully pushed TTSSpeakFrame for: '{prompt_message}'")
        except Exception as e:
            logger.error(f"Error pushing TTSSpeakFrame for '{prompt_message}': {e}", exc_info=True)
       
            if final_goodbye_message:
                logger.warning("Proceeding to push EndFrame despite TTS error for final goodbye.")
            else:
                return True 

    if final_goodbye_message: 
        logger.info("Max silence prompts reached. Initiating call termination sequence.")
        

        logger.info("Pushing EndFrame UPSTREAM to terminate pipeline.")
        try:
            await processor.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            logger.debug("Successfully pushed EndFrame.")
        except Exception as e:
            logger.error(f"Error pushing EndFrame: {e}", exc_info=True)
           
        return False  # Stop monitoring
    
    return True  # Continue monitoring

async def main(
    room_url: str,
    token: str,
    body: dict,
):
    
    global _call_start_time, _silence_event_count, _summary_logged_flag, _active_pipeline_task
    
    # Reset state variables for each call
    _call_start_time = None
    _silence_event_count = 0
    _summary_logged_flag = False
    _active_pipeline_task = None
    
    call_id_for_logs = "N/A" # Placeholder for a call identifier
    
    
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    
    # Initialize user idle processor
    user_idle_processor = UserIdleProcessor(
        callback=handle_user_idle,
        timeout=10.0  #10 seconds of inactivity
    )
    
    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_idle_processor,  # User idle processor
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_instance, participant):
        nonlocal call_id_for_logs # To potentially update if more specific ID comes from participant
        global _call_start_time
        _call_start_time = datetime.datetime.now()
        participant_id = participant.get('id', 'UnknownParticipant')
        # Potentially refine call_id_for_logs if participant data is more specific
        # e.g., if call_id_for_logs was "N/A" and participant has a session_id that's better
        logger.info(f"First participant {participant_id} joined. Call timer started at: {_call_start_time.isoformat()}. Logging with ID: {call_id_for_logs}")
        await transport_instance.capture_participant_transcription(participant["id"])
        # Send initial greeting or context
        task.queue_frame(TTSSpeakFrame("Hello, welcome! How can I help you today?"))
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    # User leaves
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport_instance, participant, reason):
        global _active_pipeline_task
        participant_id = participant.get('user_name', participant.get('id', 'UnknownParticipant'))
        logger.debug(f"Participant {participant_id} left. Reason: {reason}")
        log_call_summary(f"Participant {participant_id} left: {reason}", call_id_for_logs)
        if _active_pipeline_task:
            await _active_pipeline_task.cancel()

    # Bot leaves
    @transport.event_handler("on_left") 
    async def on_left_handler(transport_instance, event_data=None):
        logger.info(f"Bot has left the Daily room. Event data: {event_data}")
        log_call_summary("Bot left room (pipeline ended or cancelled)", call_id_for_logs)
    
    # ------------ RUN PIPELINE ------------
    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))

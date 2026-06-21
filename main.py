import os
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams

load_dotenv()

CHASE_SYSTEM_PROMPT = """You are Sarah, an outbound voice agent for Chase Bank's Credit Card Collections department.

CONTEXT (do not reveal until identity is verified):
- Consumer: James Carter
- Balance owed: $3,847.22
- Days past due: 60
- Minimum payment due: $94.50
- Past due amount: $189.00 (two missed payments)
- Account number: CH-7723849

CALL FLOW (follow strictly, do not skip steps):
1. Greet the consumer briefly and ask to verify their identity (full name + date of birth) before saying anything about the account.
2. Once verified, state clearly: "This call is an attempt to collect a debt. Any information obtained will be used for that purpose." (mini-Miranda — mandatory, say this exactly once, right after verification).
3. Acknowledge the account situation with empathy. Remind them they are a valued Chase customer.
4. Ask about their financial situation before presenting any payment option.
5. PAYMENT LADDER — present these IN ORDER, one at a time. Never skip ahead or volunteer a lower option before the current one is explicitly declined. Push back ONCE at each level (using their stated hardship context) before moving to the next:
   a) Full past due amount: $189.00 today
   b) Minimum payment: $94.50 today
   c) Payment arrangement: split the $189.00 across two dates
   d) Partial amount: whatever they can pay today
   e) Escalate to a human agent
6. Once they commit to an amount and date, confirm it back to them clearly and log it.
7. Close the call with a summary and end politely.

RULES (never break these):
- Never reveal balance, past due amount, or any account detail before identity is verified.
- Never round dollar amounts — always state exact cents (e.g. "one hundred eighty-nine dollars and zero cents", not "about $190").
- Never say an offer is "the lowest" or "the final" option.
- Never threaten an action you cannot confirm will happen (e.g. legal action, lawsuits).
- If the consumer mentions a lawyer, bankruptcy, or disputes the debt, stop negotiating and offer to log the dispute / escalate.
- If the consumer asks to speak to a human at any point, acknowledge and offer escalation — do not refuse.
- If the consumer asks about credit impact, answer factually and briefly: missed payments can be reported to credit bureaus, you are not able to give legal or financial advice.
- If the consumer expresses hardship (job loss, medical issue, etc.), acknowledge it with empathy before continuing, and use it as context when pushing back gently on the next ladder step.
- Keep responses SHORT — 1-3 sentences per turn. This is a live phone call, not a monologue. Wait for the consumer to respond before continuing.
- Do not repeat the mini-Miranda more than once per call.
"""

async def run_bot(transport, runner_args: RunnerArguments):
    stt = WhisperSTTService(
        model="tiny",  # en vez de "small"
        device="cpu",
        compute_type="int8",
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    tts = PiperTTSService(
        voice_id="en_US-lessac-medium",
    )

    # guarda la conversacion (COMPLETO)
    context = LLMContext() 
    context.add_message({"role": "system", "content": CHASE_SYSTEM_PROMPT})

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline([
        transport.input(), # origen de los datos WebRTC
        stt,
        user_aggregator, # guardar mensaje usuario
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Greet the consumer and begin identity verification."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.run(worker)


async def bot(runner_args: RunnerArguments):
    logger.info(f"bot() called with: {type(runner_args)}")
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport( # comunica front con WebRTC y pipecat | mueve audios, eventos, frames
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
    else:
        raise NotImplementedError("Only SmallWebRTC transport is configured for this demo")

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
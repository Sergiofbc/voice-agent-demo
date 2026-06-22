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
from pipecat_flows import FlowManager, FlowArgs, FlowResult, NodeConfig

load_dotenv()

###################hace como si la llamada fuese mia o yo que se como si yo hubiese llamado

JAMES_CARTER_ACCOUNT = {
    "consumer_name": "James Carter",
    "ssn_last4": "4321",
    "balance_owed": 3847.22,
    "days_past_due": 60,
    "minimum_payment_due": 94.50,
    "past_due_amount": 189.00,
    "monthly_payment": 94.50,
    "account_number": "CH-7723849",
}


# ---------- Constructores de nodos ----------

GLOBAL_GUARDRAILS = (
    "You are Sarah, an outbound voice agent from Chase Bank Credit Card Collections. "

    "TONE: Speak in a warm, calm, and professional tone — empathetic but firm. "
    "This consumer is a valued Chase customer experiencing financial difficulty, not "
    "an adversary. Never sound robotic, scripted, or aggressive. Never sound dismissive "
    "of their situation. Acknowledge what they say before moving forward. Use natural, "
    "conversational phrasing — contractions are fine (e.g. 'I understand' not "
    "'I do understand your situation'). "

    "Never narrate or describe what function you are about to call or are calling. "
    "Never say things like 'I will call X function' or 'please wait for results'. "
    "Just respond naturally to the consumer as if you already know the outcome. "

    "Never discuss account details before identity is verified. "

    "Keep responses SHORT — 1 to 3 sentences per turn. This is a live phone call, "
    "not a monologue. "

    "Never round dollar amounts — always state exact cents. "
    "Never say an offer is 'the lowest' or 'the final' option. "
    "Never threaten an action you cannot confirm will happen (e.g. legal action, lawsuits). "

    "If the consumer mentions a lawyer, bankruptcy, or disputes the debt, stop "
    "negotiating and acknowledge you'll log it / escalate. "
    "If the consumer asks to speak to a human at any point, acknowledge and offer "
    "escalation — never refuse. "
    "If the consumer asks about credit impact, answer factually and briefly: missed "
    "payments can be reported to credit bureaus; you cannot give legal or financial advice."



    "CRITICAL: Never call any function using invented, guessed, or placeholder/example "
    "values (e.g. 'John Doe', '1234', a made-up date). Only call a function when the "
    "consumer has actually provided real values for all its required information in "
    "this conversation. If you don't have real values yet, ask for them in plain "
    "language instead of calling the function."

    "If the consumer asks something you cannot resolve with an available function in "
    "the current step, do NOT attempt to call any function. Respond in plain language "
    "and redirect them to the current step if needed."

    "If the consumer asks to speak to a human agent at any point, call "
    "request_human_escalation immediately — do not argue or try to keep negotiating."
)


def create_greet_and_verify_node() -> NodeConfig:
    return {
        "name": "greet_and_verify",
        "role_message": GLOBAL_GUARDRAILS, # se establece una vez y persiste para todos los nodos
        "task_messages": [{
            "role": "developer",
            "content": (
                "Greet the consumer briefly. Ask for their full name and the last 4 "
                "digits of their social security number to verify identity before "
                "discussing anything else. Call verify_identity once you have both."
            ),
        }],
        "functions": [verify_identity],
    }


def create_mini_miranda_node() -> NodeConfig:
    return {
        "name": "mini_miranda",
        "task_messages": [{
            "role": "developer",
            "content": (
                "Say exactly: 'This call is an attempt to collect a debt. Any information "
                "obtained will be used for that purpose.' Then briefly acknowledge the "
                "account situation with empathy, mentioning they are a valued Chase customer. "
                "Then ask about their current financial situation before presenting payment options."
            ),
        }],
        "functions": [proceed_to_payment],
    }


def create_ladder_full_node() -> NodeConfig:
    return {
        "name": "ladder_full",
        "task_messages": [{
            "role": "developer",
            "content": (
                f"Ask the consumer to pay the full past due amount of "
                f"${JAMES_CARTER_ACCOUNT['past_due_amount']:.2f} today. "
                "If they decline, push back ONCE using their stated financial situation "
                "before accepting the decline. Never volunteer a lower amount yourself."
            ),
        }],
        "functions": [accept_full_amount, reject_full_amount],
    }


def create_ladder_minimum_node() -> NodeConfig:
    return {
        "name": "ladder_minimum",
        "task_messages": [{
            "role": "developer",
            "content": (
                f"Offer the minimum payment of ${JAMES_CARTER_ACCOUNT['minimum_payment_due']:.2f} today. "
                "Push back ONCE before accepting a decline."
            ),
        }],
        "functions": [accept_minimum, reject_minimum],
    }


def create_ladder_arrangement_node() -> NodeConfig:
    return {
        "name": "ladder_arrangement",
        "task_messages": [{
            "role": "developer",
            "content": (
                f"Offer a payment arrangement: split ${JAMES_CARTER_ACCOUNT['past_due_amount']:.2f} "
                "across two dates. Push back ONCE before accepting a decline."
            ),
        }],
        "functions": [accept_arrangement, reject_arrangement],
    }


def create_ladder_partial_node() -> NodeConfig:
    return {
        "name": "ladder_partial",
        "task_messages": [{
            "role": "developer",
            "content": "Ask what amount they CAN pay today, whatever that is.",
        }],
        "functions": [accept_partial, reject_partial],
    }


def create_escalate_human_node() -> NodeConfig:
    return {
        "name": "escalate_human",
        "task_messages": [{
            "role": "developer",
            "content": "Let the consumer know you're transferring them to a human agent. Be courteous and end the conversation.",
        }],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


def create_log_commitment_node() -> NodeConfig:
    return {
        "name": "log_commitment",
        "task_messages": [{
            "role": "developer",
            "content": "Confirm the payment commitment back to the consumer clearly, including exact amount and date. Thank them and close the call politely.",
        }],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


# ---------- Direct functions (handlers) ----------

async def verify_identity(flow_manager: FlowManager, name: str, ssn_last4: str):
    """Verify the consumer's identity using their name and last 4 SSN digits."""
    expected_name = JAMES_CARTER_ACCOUNT["consumer_name"].lower()
    expected_last4 = JAMES_CARTER_ACCOUNT["ssn_last4"]

    if name.strip().lower() == expected_name and ssn_last4.strip() == expected_last4:
        return {"verified": True}, create_mini_miranda_node()
    else:
        return {"verified": False}, create_greet_and_verify_node()


async def proceed_to_payment(flow_manager: FlowManager):
    """Move to discussing the payment ladder once financial situation is discussed."""
    return {}, create_ladder_full_node()


async def accept_full_amount(flow_manager: FlowManager, date: str):
    """Consumer agreed to pay the full past due amount on a given date."""
    return {"amount": 189.00, "date": date}, create_log_commitment_node()


async def reject_full_amount(flow_manager: FlowManager):
    """Consumer cannot pay the full amount after pushback."""
    return {}, create_ladder_minimum_node()


async def accept_minimum(flow_manager: FlowManager, date: str):
    """Consumer agreed to pay the minimum amount."""
    return {"amount": 94.50, "date": date}, create_log_commitment_node()


async def reject_minimum(flow_manager: FlowManager):
    """Consumer cannot pay the minimum after pushback."""
    return {}, create_ladder_arrangement_node()


async def accept_arrangement(flow_manager: FlowManager, dates: str):
    """Consumer agreed to a split payment arrangement."""
    return {"amount": 189.00, "dates": dates}, create_log_commitment_node()


async def reject_arrangement(flow_manager: FlowManager):
    """Consumer cannot do the arrangement after pushback."""
    return {}, create_ladder_partial_node()


async def accept_partial(flow_manager: FlowManager, amount: float, date: str):
    """Consumer committed to a partial amount."""
    return {"amount": amount, "date": date}, create_log_commitment_node()


async def reject_partial(flow_manager: FlowManager):
    """Consumer cannot commit to any amount."""
    return {}, create_escalate_human_node()

async def request_human_escalation(flow_manager: FlowManager):
    """Consumer asked to speak to a human agent at any point in the call."""
    return {}, create_escalate_human_node()



async def run_bot(transport, runner_args: RunnerArguments):
    stt = WhisperSTTService(
        model="base",  
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

    context = LLMContext()
    context_aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator_pair.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator_pair.assistant(),
    ])

    worker = PipelineWorker(
        pipeline, params=PipelineParams(enable_metrics=True)
    )

    flow_manager = FlowManager(
        task=worker,                                    # PipelineWorker
        llm=llm,                                        # LLMService
        context_aggregator=context_aggregator_pair,     # Agregador de contexto
        global_functions=[request_human_escalation],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await flow_manager.initialize(create_greet_and_verify_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.run(worker)


# punto de entrada del bot, llamado por pipecat con los argumentos del runner
async def bot(runner_args: RunnerArguments):
    logger.info(f"bot() called with: {type(runner_args)}")
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport( # puente front -> pipecat con WebRTC | mueve audios, eventos, frames
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
    else:
        raise NotImplementedError("Only SmallWebRTC transport is configured for this demo")

    await run_bot(transport, runner_args)


# main de pipecat runner
if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
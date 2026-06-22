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


JAMES_CARTER_ACCOUNT = {
    "consumer_name": "James Carter",
    "social_security_number_last4": "4321",
    "balance_owed": 3847.22,
    "days_past_due": 60,
    "minimum_payment_due": 94.50,
    "past_due_amount": 189.00,
    "monthly_payment": 94.50,
    "account_number": "CH-7723849",
}

# Pre-formatted account strings to ensure consistency
ACCOUNT_SUMMARY = (
    f"Your Chase credit card ending in 3849 is currently 60 days past due. "
    f"Your total balance is $3,847.22. The past due amount is $189.00, "
    f"which is two missed payments of $94.50 each."
)

# ---------- Guardrails ----------

GLOBAL_GUARDRAILS = (
    "You are Sarah, an outbound collections agent from Chase Bank Credit Card Services. "
    
    "CRITICAL: You are making an OUTBOUND call. You called THEM about their delinquent account. "
    "Do NOT ask 'how can I help you' or 'what can I help you with'. "
    
    "CRITICAL: NEVER reveal account details until AFTER identity verification. "
    "Before verification, only say you're calling from Chase about their credit card account. "
    
    f"IMPORTANT ACCOUNT INFO - Use these exact values when discussing the account: "
    f"Balance: $3,847.22. Past due: $189.00. Minimum payment: $94.50. "
    f"Card ending in: 3849. Days past due: 60. "
    f"NEVER make up different numbers. NEVER try to 'look up' or 'check' the balance - "
    f"you already know it. Just state the values above. "
    
    "TONE: Warm, calm, professional, empathetic but firm. "
    "Keep responses SHORT - 1 to 3 sentences. Use contractions naturally. "
    
    "COMPLIANCE: "
    "- Mini-Miranda: 'This call is an attempt to collect a debt. Any information obtained will be used for that purpose.' "
    "- Never round amounts - always state exact cents "
    "- Never say 'lowest' or 'final' option "
    "- Never threaten legal action "
    "- If consumer mentions lawyer, bankruptcy, or disputes debt: escalate "
    "- If consumer asks for human: call request_human_escalation immediately "
    "- Credit impact: missed payments can be reported to credit bureaus, you cannot give legal/financial advice "
    
    "CRITICAL: Only call functions that exist. Available functions: "
    "verify_identity, proceed_to_payment, accept_full_amount, reject_full_amount, "
    "accept_minimum, reject_minimum, accept_arrangement, reject_arrangement, "
    "accept_partial, reject_partial, request_human_escalation. "
    "Do NOT invent functions like 'get_account_balance' or 'check_balance'. "
    "You already have all the account information you need."
)


# ---------- Nodes ----------

def create_greet_and_verify_node() -> NodeConfig:
    return {
        "name": "greet_and_verify",
        "role_message": GLOBAL_GUARDRAILS,
        "task_messages": [{
            "role": "developer",
            "content": (
                "STEP 1: Identity verification.\n"
                "Say: 'Hello, this is Sarah calling from Chase Bank Credit Card Services. "
                "Am I speaking with James Carter?'\n"
                "DO NOT mention any account numbers, balances, or amounts.\n"
                "If yes: 'For security purposes, I need to verify your identity. "
                "Can you provide the last 4 digits of your Social Security number?'\n"
                "If they ask what this is about: 'I'll explain once I verify your identity.'\n"
                "Once you have name AND SSN last 4, call verify_identity.\n"
                "If verification fails, ask them to try again. After 2 failures, call request_human_escalation."
            ),
        }],
        "functions": [verify_identity],
    }


def create_verification_failed_node() -> NodeConfig:
    return {
        "name": "verification_failed",
        "task_messages": [{
            "role": "developer",
            "content": (
                "Say: 'I'm unable to verify your identity. For your security, I'll transfer you to a representative.' "
                "Then call request_human_escalation."
            ),
        }],
        "functions": [request_human_escalation],
    }


def create_mini_miranda_and_situation_node() -> NodeConfig:
    return {
        "name": "mini_miranda",
        "task_messages": [{
            "role": "developer",
            "content": (
                "STEP 2: Mini-Miranda disclosure AND account details AND situation probe.\n"
                "You MUST say ALL of the following in order:\n"
                "1. 'Thank you for verifying, Mr. Carter.'\n"
                "2. 'This call is an attempt to collect a debt. Any information obtained will be used for that purpose.'\n"
                f"3. 'I'm calling because your Chase credit card ending in 3849 is 60 days past due. "
                f"Your balance is $3,847.22 with a past due amount of $189.00 from two missed payments "
                f"of $94.50 each.'\n"
                "4. 'I want to understand your situation. Can you tell me what's been going on financially?'\n"
                "After they respond, acknowledge their situation with empathy, then call proceed_to_payment."
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
                f"STEP 3A: Full past due amount - $189.00.\n"
                f"Say: 'To bring your account current, we'd need the full past due amount of $189.00. "
                f"Would you be able to pay that today?'\n"
                "If YES: Ask for date, call accept_full_amount.\n"
                "If NO: Push back ONCE using their situation. "
                "'I understand [situation], but is there any way to manage the $189.00?'\n"
                "If still no, call reject_full_amount.\n"
                "Do NOT mention other options yet."
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
                f"STEP 3B: Minimum payment - $94.50.\n"
                f"Say: 'I understand. We could accept the minimum payment of $94.50 "
                f"to prevent further delinquency. Would you be able to pay that today?'\n"
                "If YES: Ask for date, call accept_minimum.\n"
                "If NO: Push back ONCE: 'This would cover one missed payment and stop late fees. "
                "Can you manage $94.50?'\n"
                "If still no, call reject_minimum.\n"
                "Do NOT mention payment arrangements yet."
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
                f"STEP 3C: Payment arrangement - split $189.00 into two payments of $94.50.\n"
                f"Say: 'We can split the $189.00 into two payments of $94.50. "
                f"Would that work for you?'\n"
                "If YES: Ask for two dates, call accept_arrangement.\n"
                "If NO: Push back ONCE: 'This would prevent charge-off and protect your credit. "
                "Can we make this work?'\n"
                "If still no, call reject_arrangement.\n"
                "Do NOT mention partial payments yet."
            ),
        }],
        "functions": [accept_arrangement, reject_arrangement],
    }


def create_ladder_partial_node() -> NodeConfig:
    return {
        "name": "ladder_partial",
        "task_messages": [{
            "role": "developer",
            "content": (
                "STEP 3D: Partial payment - whatever they can pay.\n"
                "Say: 'Any payment would show good faith. What amount can you commit to today?'\n"
                "If they give ANY amount > $0 and a date, call accept_partial.\n"
                "If they say nothing, push back ONCE: 'Even $25 or $50 would help. Is there any amount?'\n"
                "If still nothing, call reject_partial."
            ),
        }],
        "functions": [accept_partial, reject_partial],
    }


def create_escalate_human_node() -> NodeConfig:
    return {
        "name": "escalate_human",
        "task_messages": [{
            "role": "developer",
            "content": (
                "Say: 'I'm transferring you to a Chase representative. "
                "Please hold. Thank you, Mr. Carter.' End conversation."
            ),
        }],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


def create_log_commitment_node() -> NodeConfig:
    return {
        "name": "log_commitment",
        "task_messages": [{
            "role": "developer",
            "content": (
                "Confirm: 'Thank you, Mr. Carter. I've noted your payment of $[amount] for [date]. "
                "You'll receive confirmation. We appreciate your commitment. Have a good day.' "
                "End conversation."
            ),
        }],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


# ---------- Functions ----------

verification_attempts = 0

async def verify_identity(flow_manager: FlowManager, name: str, ssn_last4: str):
    """Verify the consumer's identity."""
    global verification_attempts
    
    expected_name = JAMES_CARTER_ACCOUNT["consumer_name"].lower()
    expected_last4 = JAMES_CARTER_ACCOUNT["ssn_last4"]
    
    name_clean = name.strip().lower()
    ssn_clean = ssn_last4.strip()
    
    logger.info(f"Verify: name='{name_clean}' ssn='{ssn_clean}'")
    
    if name_clean == expected_name and ssn_clean == expected_last4:
        verification_attempts = 0
        logger.info("Verification SUCCESS")
        return {"verified": True}, create_mini_miranda_and_situation_node()
    else:
        verification_attempts += 1
        logger.warning(f"Verification FAILED (attempt {verification_attempts})")
        if verification_attempts >= 2:
            verification_attempts = 0
            return {"verified": False}, create_verification_failed_node()
        else:
            return {"verified": False}, create_greet_and_verify_node()


async def proceed_to_payment(flow_manager: FlowManager):
    """Move to payment ladder."""
    return {}, create_ladder_full_node()


async def accept_full_amount(flow_manager: FlowManager, date: str):
    return {"amount": 189.00, "date": date}, create_log_commitment_node()


async def reject_full_amount(flow_manager: FlowManager):
    return {}, create_ladder_minimum_node()


async def accept_minimum(flow_manager: FlowManager, date: str):
    return {"amount": 94.50, "date": date}, create_log_commitment_node()


async def reject_minimum(flow_manager: FlowManager):
    return {}, create_ladder_arrangement_node()


async def accept_arrangement(flow_manager: FlowManager, dates: str):
    return {"amount": 189.00, "dates": dates}, create_log_commitment_node()


async def reject_arrangement(flow_manager: FlowManager):
    return {}, create_ladder_partial_node()


async def accept_partial(flow_manager: FlowManager, amount: float, date: str):
    return {"amount": amount, "date": date}, create_log_commitment_node()


async def reject_partial(flow_manager: FlowManager):
    return {}, create_escalate_human_node()


async def request_human_escalation(flow_manager: FlowManager):
    logger.info("Human escalation requested")
    return {}, create_escalate_human_node()


async def run_bot(transport, runner_args: RunnerArguments):
    stt = WhisperSTTService(
        model="small",
        device="cpu",
        compute_type="int8",
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        params={
            "temperature": 0.2,
        }
    )

    tts = PiperTTSService(
        voice_id="en_US-lessac-medium",
    )

    context = LLMContext()
    context_aggregator_pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
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
        task=worker,
        llm=llm,
        context_aggregator=context_aggregator_pair,
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


async def bot(runner_args: RunnerArguments):
    logger.info(f"bot() called with: {type(runner_args)}")
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport(
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
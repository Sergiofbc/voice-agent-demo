# Voice AI Debt Collection Agent with Pipecat

## Overview

This project implements a real-time voice AI agent for debt collection scenarios using Pipecat.

The system conducts a simulated outbound collections call following a predefined payment negotiation flow while enforcing compliance requirements such as identity verification and Mini-Miranda disclosure.

The solution combines local speech processing with cloud-hosted large language models in a hybrid architecture designed for low-latency voice interactions.

---

## Architecture

The application follows the classic conversational AI **"sandwich" architecture** (commonly referred to in frameworks such as LangChain):

```text
User Audio
    ↓
Whisper STT
    ↓
Groq LLM (Llama 3.3 70B)
    ↓
Piper TTS
    ↓
Audio Response
```

Pipecat orchestrates the entire pipeline and manages real-time communication through WebRTC.

### Technology Stack

| Component                | Technology                     |
| ------------------------ | ------------------------------ |
| Orchestration            | Pipecat                        |
| Transport Layer          | WebRTC                         |
| Speech-to-Text           | Whisper                        |
| Voice Activity Detection | Silero VAD                     |
| LLM                      | Groq (Llama 3.3 70B Versatile) |
| Text-to-Speech           | Piper                          |
| Runtime                  | Python                         |

---

## Why Pipecat?

Pipecat was selected because it is a native orchestration framework for multimodal AI agents.

Instead of manually implementing:

* Audio streaming
* Event handling
* Interruptions
* Buffer management
* Real-time transport
* Context propagation

Pipecat provides these capabilities as reusable building blocks.

Additionally, Pipecat supports a transition from a traditional procedural architecture toward a declarative state-machine approach through Pipecat Flows.

This becomes particularly valuable for debt collection conversations where a predefined negotiation sequence must be respected.

---

## Hybrid Architecture

The project uses a hybrid local/cloud deployment model.

### Local Components

* Whisper STT
* Piper TTS
* Silero VAD

Running these services locally reduces latency and minimizes external dependencies for audio processing.

### Cloud Components

* Groq LLM API

The selected language model is Llama 3.3 70B.

A model of this size is not practical to run locally on consumer hardware, therefore inference is delegated to Groq's optimized infrastructure while maintaining low response latency.

---

## Conversation Design

The agent follows a strict debt collection workflow.

### 1. Identity Verification

The consumer must verify:

* Full name
* Last four digits of SSN

No account information is revealed before successful verification.

### 2. Mini-Miranda Disclosure

Immediately after verification:

> This call is an attempt to collect a debt. Any information obtained will be used for that purpose.

### 3. Situation Assessment

The agent gathers information regarding:

* Financial hardship
* Employment status
* Consumer circumstances

### 4. Payment Ladder

Negotiation follows a structured sequence:

1. Full past-due amount
2. Minimum payment
3. Payment arrangement
4. Partial payment
5. Human escalation

The agent cannot skip levels.

### 5. Commitment Confirmation

The agreed amount and payment date are confirmed back to the consumer.

### 6. Call Closure

A concise summary is provided before ending the conversation.

---

## Compliance Strategy

### FDCPA Compliance

The flow was designed to support debt collection regulations by enforcing:

* Identity verification before disclosure
* Mandatory Mini-Miranda statement
* No legal threats
* No misleading statements
* Escalation when disputes arise
* Escalation when bankruptcy or attorney representation is mentioned

### Quality Controls

The system includes:

* Structured conversation flow
* Consistent payment negotiation process
* Response brevity for voice interactions
* Final call summary
* Post-call survey support (future enhancement)

---

## Context Management

Conversation history is preserved using Pipecat's context aggregation system:

```python
context = LLMContext()

user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
    ),
)
```

The aggregators automatically store:

* User utterances
* Assistant responses
* System instructions

allowing the model to maintain conversational continuity across the entire call.

---

## Current Architecture vs Future Architecture

### Current Implementation

```text
Whisper
   ↓
Context Aggregator
   ↓
Single Prompt
   ↓
Groq LLM
   ↓
Piper
```

### Planned Architecture (Pipecat Flows)

```text
Identity Verification State
           ↓
Mini-Miranda State
           ↓
Financial Assessment State
           ↓
Payment Ladder State
           ↓
Commitment State
           ↓
Closure State
```

Moving to Pipecat Flows would provide stronger control over compliance requirements and business logic while reducing prompt complexity.

---

## Running the Project

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

### Installation

```bash
pip install -r requirements.txt
```

### Start the Agent

```bash
python main.py
```

---

## Future Improvements

### Pipecat Flows Migration

Convert the current prompt-driven implementation into a declarative state machine.

Expected benefits:

* Stronger compliance guarantees
* Easier maintenance
* Better observability
* Reduced prompt complexity

### Database Integration

Move customer information and account data from hardcoded prompts into a database-backed solution.

Potential use cases:

* Customer lookup
* Account retrieval
* Payment history
* Call logging
* Dynamic prompt generation

### Telephony Integration

Integrate providers such as:

* Twilio
* Daily
* SIP-based telephony systems

This would allow the agent to place and receive real phone calls.

### Human Handoff

Enable seamless escalation to human collection agents when required.


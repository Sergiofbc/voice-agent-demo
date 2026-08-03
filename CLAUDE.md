# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of **Pipecat**-based real-time voice AI agent prototypes. There is no single application — the repo contains two independent bot lineages at different stages of iteration, plus a RAG ingestion pipeline. There is no test suite, linter, or build system configured anywhere in the repo; these are demo/prototype scripts.

## The two bot lineages

1. **Debt-collection bot** (root: `agent.py`, `agentDeep.py`, `botLocal.py`, `deploy/bot.py`) — English-speaking "Sarah" from Chase Bank Credit Card Collections. Fixed demo consumer (`JAMES_CARTER_ACCOUNT`) hardcoded in each file. Stack: Whisper (local STT) → Groq Llama 3.3 70B → Piper (local TTS).
   - `botLocal.py`: earliest iteration — a single plain system prompt (`CHASE_SYSTEM_PROMPT`), no Flows, one `LLMContext` fed straight to the LLM each turn.
   - `agent.py` / `agentDeep.py`: rewritten on **Pipecat Flows** — the call is a state machine of `NodeConfig` nodes (`greet_and_verify → mini_miranda → ladder_full → ladder_minimum → ladder_arrangement → ladder_partial → log_commitment`, with `escalate_human` reachable from any step). `agentDeep.py` is the more defensive version: it enforces exact account figures in the prompt, tracks `verification_attempts` to force escalation after 2 failed identity checks, and restricts which function names the LLM is allowed to call. `agent.py` is the lighter/earlier variant of the same flow.
   - `deploy/bot.py`: the Flows version packaged for shipping (see Deployment below).

2. **Tech-support bot ("soporte técnico ofimático")** (`soporte-tecnico/`, `soporte-tecnico-deploy/`) — Spanish-speaking helpdesk agent for Excel/Teams/SharePoint/Microsoft-account issues. Stack: ElevenLabs (STT + TTS) → Groq Llama 3.3 70B, with a Pinecone-backed RAG tool.
   - `soporte-tecnico/bot.py`: Flows-based (`saludo_identificacion → diagnostico → cierre`), shared RAG resources loaded via a dataclass (`AppResources`) passed into `PipelineWorker`.
   - `soporte-tecnico/bot2.py`: the more refined/current version of the same flow — shared RAG resources live on `flow_manager.state["recursos_rag"]` instead of a dataclass, uses `@tool_options(cancel_on_interruption=False)` on tools that shouldn't be cut off mid-call, and switches transport via `create_transport(runner_args, transport_params)` keyed by the `TRANSPORT` env var (`"local"` → WebRTC, `"daily"` → Daily/Pipecat Cloud). **Read the module docstring and inline comments in this file first** — it documents the Flows function-signature convention (`flow_manager: FlowManager` as first arg, not `FunctionCallParams`) and the node-function vs edge-function return convention (`result` vs `(result, next_node)`).
   - `soporte-tecnico-deploy/bot.py`: a simplified, non-Flows/non-RAG variant (plain `HELPDESK_SYSTEM_PROMPT`) — used to test the ElevenLabs STT/TTS pipeline in isolation without Flows/RAG complexity.
   - `soporte-tecnico-deploy/test11leven.py`: standalone script hitting the ElevenLabs TTS REST API directly (writes `test.mp3`) — a quick way to sanity-check the ElevenLabs API key/voice ID outside the full pipeline.

Because both lineages evolved by copy-and-modify rather than shared modules, **duplicated logic across files is expected** — when fixing a bug, check whether the same bug exists in the sibling file(s) for that lineage before assuming one fix is enough.

## Core Pipecat architecture (applies to every bot file)

Every `bot.py`/`agent*.py` follows the same shape:

```
transport.input() → STT → user_aggregator → LLM → TTS → transport.output() → assistant_aggregator
```

- **`Pipeline`** declares this processing order; **`PipelineWorker`** actually runs it and reacts to frames/events; **`WorkerRunner`** keeps the worker alive across the process lifetime.
- **`Transport`** is the boundary with the outside world (WebRTC locally, Daily in Pipecat Cloud). `@transport.event_handler("on_client_connected"/"on_client_disconnected")` callbacks are invoked by Pipecat itself, never called manually.
- **`LLMContext`** is the conversation history (`List[{role, content}]`). `LLMContextAggregatorPair` keeps it in sync: the user aggregator appends STT output as `role: user`, the assistant aggregator appends LLM output as `role: assistant`.
- The entry point Pipecat calls is always `async def bot(runner_args: RunnerArguments)`, which builds/selects the `Transport` and delegates to `run_bot(transport, runner_args)`. `if __name__ == "__main__": from pipecat.runner.run import main; main()` is the CLI entry point (`python bot.py -t webrtc`).
- See `context_first_agent.txt` for a longer first-person walkthrough of this architecture (in Spanish) — useful background before touching any bot file.

### Flows vs. plain-prompt pattern

Files using `pipecat_flows` (`agent.py`, `agentDeep.py`, `soporte-tecnico/bot.py`, `soporte-tecnico/bot2.py`) model the call as a graph of `NodeConfig` objects (`role_message` = persistent persona/guardrails set once, `task_messages` = per-node instructions, `functions` = tools available in that node). Tool functions return either just a result dict (node function, stays put) or `(result, next_node)` (edge function, transitions the flow). Files without Flows (`botLocal.py`, `soporte-tecnico-deploy/bot.py`) just mutate one big system-prompt-driven `LLMContext` and re-run the LLM on each turn via `worker.queue_frames([LLMRunFrame()])`.

### RAG (soporte-tecnico only)

- `soporte-tecnico/KB.json` is the source catalog: `{producto: {pregunta_en_lenguaje_natural: url_soporte_microsoft}}`.
- `soporte-tecnico/ingesta.py` downloads each URL, strips boilerplate with `trafilatura`, chunks by markdown `##` headings (falling back to word-count chunking for oversized sections), embeds locally with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (no API cost), and upserts into the Pinecone index `soporte-ofimatico`. Run it whenever `KB.json` changes: `python soporte-tecnico/ingesta.py` (needs `PINECONE_API_KEY`).
- At runtime, the `buscar_en_base_conocimiento` tool in `bot.py`/`bot2.py` embeds the user's query the same way, queries the same index (`top_k=3`), and only trusts a match if `score >= UMBRAL_CONFIANZA_RAG` (0.55). Below that threshold it must tell the LLM explicitly not to invent a solution — this guardrail is load-bearing; don't lower the threshold or drop the "don't invent" instruction without a reason.

## Environment variables

Each bot directory has its own `.env` (all gitignored):
- Root `.env`: `GROQ_API_KEY` (debt-collection bots).
- `soporte-tecnico/.env`, `soporte-tecnico-deploy/.env`: `GROQ_API_KEY`, `ELEVENLABS_API_KEY`, `PINECONE_API_KEY` (soporte-tecnico only), `TRANSPORT` (`local` or `daily`).
- `TRANSPORT` gates a conditional import: `from pipecat.transports.daily.transport import DailyParams` is only imported when `TRANSPORT == "daily"`, so Daily-specific dependencies aren't required for local WebRTC runs.

## Running locally

```bash
python <bot_file>.py -t webrtc
```
The `-t` flag and other CLI args are parsed by Pipecat's own `main()` (`pipecat.runner.run`), not by the bot files themselves.

## Deployment (`deploy/`)

Packages the Chase collections bot for **Pipecat Cloud**:
- `deploy/Dockerfile` builds `FROM dailyco/pipecat-base:latest`, installs `deploy/requirements.txt`, copies `deploy/bot.py`.
- `deploy/pcc-deploy.toml` names the agent `chase-bank-bot`, references image `basergi/chase-bank-bot:0.8` and secret set `chase-secrets`.
- `deploy/requirements.txt` pins: `groq`, `faster-whisper`, `pipecat-ai[silero,piper,daily]`, `python-dotenv`, `pipecatcloud`.

There is no equivalent deploy packaging yet for the soporte-tecnico bots — `soporte-tecnico-deploy/` currently only holds a simplified test variant of that bot, not Docker/deploy config.

## Compliance/behavioral constraints baked into the collections bot prompts

These are product requirements, not incidental prompt text — preserve them when editing `agent.py`/`agentDeep.py`/`botLocal.py`/`deploy/bot.py`:
- No account details (balance, past-due amount, etc.) before identity verification succeeds.
- Mini-Miranda disclosure ("This call is an attempt to collect a debt...") exactly once, right after verification.
- Payment ladder must be walked in strict order (full → minimum → arrangement → partial → human escalation), pushing back at most once per level, never volunteering a lower option early.
- Never round dollar amounts; never call an offer "the lowest"/"the final"; never threaten unconfirmable actions (e.g. legal action).
- Immediate escalation on: request for a human, mention of a lawyer/bankruptcy, or a debt dispute.
- `testPlan.md` is the manual QA script enumerating these scenarios end-to-end — use it to sanity-check behavioral changes to the collections flow.

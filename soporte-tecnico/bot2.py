"""
Bot de soporte técnico ofimático.

Arquitectura:
    STT  -> ElevenLabs (Scribe realtime)
    LLM  -> Groq (Llama 3.3 70B, compatible con function calling y Flows)
    TTS  -> ElevenLabs (Flash v2.5, voz con acento colombiano)
    Orquestación -> Pipecat + Pipecat Flows (3 nodos macro)
    Conocimiento -> Pinecone (RAG) vía function calling
    Transporte -> aislado automáticamente por create_transport()
                  (WebRTC local en desarrollo, Daily en Pipecat Cloud)

Correr en local:
    python bot.py -t webrtc

Desplegar en Pipecat Cloud:
    el mismo archivo, sin cambios.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat_flows import FlowManager, NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
TRANSPORT = os.getenv("TRANSPORT", "local")

if TRANSPORT == "daily":
    from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config de RAG
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINECONE_INDEX_NAME = "soporte-ofimatico"
UMBRAL_CONFIANZA_RAG = 0.55  # score mínimo de similitud para considerar un match confiable

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(), ##########################
    ),
}


def cargar_recursos_rag():
    """Carga el modelo de embeddings y la conexión a Pinecone una sola vez por sesión."""
    logger.info("Cargando modelo de embeddings y conexión a Pinecone...")
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    indice = pc.Index(PINECONE_INDEX_NAME)
    return {"embedding_model": modelo, "pinecone_index": indice}


# ---------------------------------------------------------------------------
# Funciones de Flows
#
# IMPORTANTE: en Pipecat Flows, el primer argumento de una función directa es
# `flow_manager: FlowManager`, NO `FunctionCallParams` (ese es el patrón del
# function-calling "base" de Pipecat sin Flows). Los recursos compartidos
# (Pinecone, el modelo de embeddings) viven en `flow_manager.state`, que se
# inicializa una vez en run_bot().
#
# - Una "node function" (no cambia de nodo) simplemente retorna el resultado.
# - Una "edge function" (transiciona a otro nodo) retorna (resultado, next_node).
# ---------------------------------------------------------------------------

PERSONALIDAD_BOT = """
Eres un agente de soporte técnico ofimático (Excel, Microsoft Teams, SharePoint, cuentas Microsoft)
para una mesa de ayuda. Hablas español neutro/colombiano, de forma cálida y conversacional,
como un compañero de trabajo que ayuda por teléfono, NO como si estuvieras leyendo un manual.
 
Reglas de estilo (esto es una llamada de voz, muy importante):
- Responde en turnos cortos: máximo 2-3 frases por turno.
- Guía paso a paso, esperando confirmación del usuario entre cada paso. Nunca sueltes
  todo un procedimiento de una vez.
- No leas contenido técnico literalmente. Reformúlalo con tus propias palabras.
- Expande abreviaturas técnicas para que se escuchen bien en voz
  (ej. di "control zeta" en vez de leer el símbolo "Ctrl+Z").
- Nunca inventes procedimientos de solución. Si no tienes información confiable
  de la base de conocimiento, dilo y pide más detalles, o escala el caso.
- Cuando necesites usar una función (identificar_usuario, buscar_en_base_conocimiento,
  marcar_resuelto, escalar_a_humano), invócala usando el mecanismo de function calling
  de la API. NUNCA escribas el nombre de la función ni su sintaxis como parte de tu
  respuesta de texto — eso se escucharía en voz alta, lo cual está prohibido.
"""


@tool_options(cancel_on_interruption=False)
async def identificar_usuario(flow_manager: FlowManager, nombre: str):
    """Registra el nombre de la persona que llama y avanza a la etapa de diagnóstico.

    Args:
        nombre: Nombre completo o como se identificó la persona que llama.
    """
    # Aquí iría la consulta real a tu base de datos de usuarios (por HTTP).
    # Se deja como stub porque esa base todavía no está construida.
    logger.info(f"Usuario identificado: {nombre}")
    flow_manager.state["nombre_usuario"] = nombre
    resultado = {"identificado": True, "nombre": nombre}
    return resultado, crear_nodo_diagnostico(nombre)


async def buscar_en_base_conocimiento(flow_manager: FlowManager, consulta: str):
    """Busca en la base de conocimiento de soporte ofimático (Excel, Teams, SharePoint, cuentas Microsoft).

    Úsala cada vez que necesites un procedimiento de solución concreto,
    en vez de inventarlo de memoria.

    Args:
        consulta: La descripción del problema en las palabras del usuario, tal cual las dijo.
    """
    recursos = flow_manager.state["recursos_rag"]

    # El modelo de embeddings es síncrono; lo corremos en un hilo aparte
    # para no bloquear el loop de eventos del pipeline.
    vector = await asyncio.to_thread(
        recursos["embedding_model"].encode, consulta, normalize_embeddings=True
    )
    respuesta = await asyncio.to_thread(
        recursos["pinecone_index"].query,
        vector=vector.tolist(),
        top_k=3,
        include_metadata=True,
    )

    matches = respuesta.get("matches", [])
    if not matches or matches[0]["score"] < UMBRAL_CONFIANZA_RAG:
        # Guardrail: si no hay match confiable, se lo decimos explícitamente
        # al LLM para que no invente una solución.
        return {
            "encontrado": False,
            "instruccion": (
                "No se encontró información confiable. No inventes una solución. "
                "Pide más detalles o, si ya intentaste 2 veces sin éxito, ofrece escalar."
            ),
        }

    mejor = matches[0]
    return {
        "encontrado": True,
        "contenido": mejor["metadata"]["content"],
        "producto": mejor["metadata"].get("producto"),
        "url_referencia": mejor["metadata"].get("url"),
    }


async def marcar_resuelto(flow_manager: FlowManager):
    """Llama a esta función cuando el usuario confirma explícitamente que su problema quedó resuelto."""
    return {"estado": "resuelto"}, crear_nodo_cierre(resuelto=True)


@tool_options(cancel_on_interruption=False)
async def escalar_a_humano(flow_manager: FlowManager, resumen_problema: str):
    """Escala el caso a un agente humano cuando no fue posible resolver el problema por este medio.

    Args:
        resumen_problema: Resumen breve (1-2 frases) del problema y de lo que ya se intentó.
    """
    # Aquí iría la creación real del ticket (HTTP a tu backend/CRM).
    logger.warning(f"Caso escalado: {resumen_problema}")
    return {"estado": "escalado", "resumen": resumen_problema}, crear_nodo_cierre(resuelto=False)


# ---------------------------------------------------------------------------
# Nodos: Saludo/Identificación -> Diagnóstico -> Resolución/Escalamiento
# ---------------------------------------------------------------------------

def crear_nodo_saludo() -> NodeConfig:
    return NodeConfig(
        name="saludo_identificacion",
        role_message=PERSONALIDAD_BOT,
        task_messages=[{
            "role": "developer",
            "content": (
                "Saluda brevemente y pregunta el nombre de la persona para identificarla. "
                "No preguntes nada más todavía."
            ),
        }],
        functions=[identificar_usuario],
    )


def crear_nodo_diagnostico(nombre_usuario: str) -> NodeConfig:
    return NodeConfig(
        name="diagnostico",
        task_messages=[{
            "role": "developer",
            "content": (
                f"Ya identificaste a {nombre_usuario}. Pregúntale con qué problema necesita "
                "ayuda. Usa buscar_en_base_conocimiento para encontrar la solución antes de "
                "responder con procedimientos técnicos. Guíalo paso a paso. "
                "Si el usuario confirma que el problema quedó resuelto, llama a marcar_resuelto. "
                "Si el problema persiste tras intentos razonables, o el usuario lo pide, "
                "llama a escalar_a_humano."
            ),
        }],
        functions=[buscar_en_base_conocimiento, marcar_resuelto, escalar_a_humano],
    )


def crear_nodo_cierre(resuelto: bool) -> NodeConfig:
    mensaje = (
        "El problema quedó resuelto. Agradece a la persona y despídete brevemente."
        if resuelto else
        "El caso fue escalado a un agente humano. Avísale que alguien del equipo la "
        "contactará pronto, agradece su paciencia y despídete brevemente."
    )
    return NodeConfig(
        name="cierre",
        task_messages=[{"role": "developer", "content": mensaje}],
        functions=[],
        post_actions=[{"type": "end_conversation"}],
    )


# ---------------------------------------------------------------------------
# Pipeline y entry point del bot
# ---------------------------------------------------------------------------

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        settings=ElevenLabsRealtimeSTTService.Settings(
            model="scribe_v2_realtime",
            language="es",
            vad_threshold=0.65,
            vad_silence_threshold_secs=0.9,
            min_speech_duration_ms=300,
            min_silence_duration_ms=300,
        ),
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY", ""),
        settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        settings=ElevenLabsTTSService.Settings(
            voice="6Gr4AVmTax1pMJO0lHRK",  # Chile: 6Gr4AVmTax1pMJO0lHRK - Colombia : b2htR0pMe28pYwCY9gnP
            model="eleven_flash_v2_5",
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=False,  # Mega Prompt para que el LLM no se quede esperando a que el usuario termine de hablar, sino que procese lo que haya dicho hasta ahora.
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # Recursos de RAG disponibles para cualquier función de Flows vía flow_manager.state
    flow_manager.state["recursos_rag"] = cargar_recursos_rag()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Cliente conectado")
        await flow_manager.initialize(crear_nodo_saludo())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Cliente desconectado")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Punto de entrada del bot — funciona igual en local y en Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
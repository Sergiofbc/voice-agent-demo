"""
Bot de soporte técnico ofimático.

Arquitectura:
    STT  -> ElevenLabs (Scribe, tiempo real)
    LLM  -> Groq (Llama 3.3 70B, compatible con function calling y Flows)
    TTS  -> ElevenLabs (Flash v2.5, voz con acento colombiano)
    Orquestación -> Pipecat + Pipecat Flows (3 nodos macro)
    Conocimiento -> Pinecone (RAG) vía function calling
    Transporte -> aislado automáticamente por create_transport()
                  (WebRTC local en desarrollo, Daily en Pipecat Cloud)

Correr en local:
    python bot.py -t webrtc

Desplegar en Pipecat Cloud:
    el mismo archivo, sin cambios (ver docs de pipecat-cloud deploy)
"""

import asyncio
from multiprocessing import context
from multiprocessing import context
import os
from dataclasses import dataclass
from unittest import runner

from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.run import main as runner_main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.pipeline.runner import WorkerRunner
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams

TRANSPORT = os.getenv("TRANSPORT", "local")

if TRANSPORT == "daily":
    from pipecat.transports.daily.transport import DailyParams

from pipecat_flows import FlowManager, NodeConfig

load_dotenv()


ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# Recursos compartidos (RAG): se cargan una sola vez y se pasan a los tools
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PINECONE_INDEX_NAME = "soporte-ofimatico"
UMBRAL_CONFIANZA_RAG = 0.55  # score mínimo de similitud para considerar un match confiable


@dataclass
class AppResources:
    """Recursos compartidos entre todos los tools del bot."""
    pinecone_index: object
    embedding_model: SentenceTransformer


def cargar_recursos() -> AppResources:
    logger.info("Cargando modelo de embeddings y conexión a Pinecone...")
    modelo = SentenceTransformer(EMBEDDING_MODEL_NAME)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    indice = pc.Index(PINECONE_INDEX_NAME)
    return AppResources(pinecone_index=indice, embedding_model=modelo)


# ---------------------------------------------------------------------------
# Tools (funciones que el LLM puede invocar)
# ---------------------------------------------------------------------------

async def identificar_usuario(flow_manager: FlowManager, nombre: str) -> tuple[dict, NodeConfig]:
    """Registra el nombre de la persona que llama y avanza a la etapa de diagnóstico.

    Args:
        nombre: Nombre completo o como se identificó la persona que llama.
    """
    # Aquí iría la consulta real a tu base de datos de usuarios (por HTTP).
    # La dejamos como stub porque aún no está construida esa base.
    logger.info(f"Usuario identificado: {nombre}")
    resultado = {"identificado": True, "nombre": nombre}
    return resultado, crear_nodo_diagnostico(nombre)


async def buscar_en_base_conocimiento(params: FunctionCallParams, consulta: str):
    """Busca en la base de conocimiento de soporte técnico ofimático (Excel, Teams, SharePoint, cuentas Microsoft).

    Úsala cada vez que necesites información concreta sobre un procedimiento de solución,
    en vez de inventar pasos de memoria.

    Args:
        consulta: La descripción del problema en las palabras del usuario, tal cual las dijo.
    """
    recursos: AppResources = params.app_resources

    # El modelo de embeddings es síncrono y puede tardar unos ms; lo corremos
    # en un hilo aparte para no bloquear el loop de eventos del pipeline.
    vector = await asyncio.to_thread(
        recursos.embedding_model.encode, consulta, normalize_embeddings=True
    )

    respuesta = await asyncio.to_thread(
        recursos.pinecone_index.query,
        vector=vector.tolist(),
        top_k=3,
        include_metadata=True,
    )

    matches = respuesta.get("matches", [])
    if not matches or matches[0]["score"] < UMBRAL_CONFIANZA_RAG:
        # Guardrail importante: si no hay un match confiable, se lo decimos
        # explícitamente al LLM para que NO invente una solución.
        await params.result_callback({
            "encontrado": False,
            "instruccion": (
                "No se encontró información confiable. No inventes una solución. "
                "Pide más detalles al usuario o, si ya intentaste 2 veces sin éxito, "
                "ofrece escalar el caso."
            ),
        })
        return

    mejor = matches[0]
    await params.result_callback({
        "encontrado": True,
        "contenido": mejor["metadata"]["content"],
        "producto": mejor["metadata"].get("producto"),
        "url_referencia": mejor["metadata"].get("url"),
    })


async def marcar_resuelto(params: FunctionCallParams) -> tuple[dict, NodeConfig]:
    """Llama a esta función cuando el usuario confirma explícitamente que su problema quedó resuelto."""
    return {"estado": "resuelto"}, crear_nodo_cierre(resuelto=True)


async def escalar_a_humano(params: FunctionCallParams, resumen_problema: str) -> tuple[dict, NodeConfig]:
    """Escala el caso a un agente humano cuando no fue posible resolver el problema por este medio.

    Args:
        resumen_problema: Resumen breve (1-2 frases) del problema y de lo que ya se intentó.
    """
    # Aquí iría la creación real del ticket (HTTP a tu backend/CRM).
    logger.warning(f"Caso escalado: {resumen_problema}")
    return {"estado": "escalado", "resumen": resumen_problema}, crear_nodo_cierre(resuelto=False)


# ---------------------------------------------------------------------------
# Nodos del flujo: Saludo/Identificación -> Diagnóstico -> Resolución/Escalamiento
# ---------------------------------------------------------------------------

PERSONALIDAD_BOT = """
Eres un agente de soporte técnico ofimático (Excel, Microsoft Teams, SharePoint, cuentas Microsoft)
para una mesa de ayuda. Hablas español neutro/colombiano, de forma cálida y conversacional,
como un compañero de trabajo que ayuda por teléfono — NO como si estuvieras leyendo un manual.

Reglas de estilo (muy importantes, esto es una llamada de voz):
- Responde en turnos cortos: máximo 2-3 frases por turno.
- Guía paso a paso, esperando confirmación del usuario entre cada paso. Nunca sueltes
  todo un procedimiento de una vez.
- No leas contenido técnico literalmente. Reformúlalo con tus propias palabras.
- Deletrea o expande abreviaturas técnicas para que se escuchen bien en voz
  (ej. "Ctrl+Z" dilo como "control zeta", no leas el símbolo "+").
- Nunca inventes procedimientos de solución. Si no tienes información confiable
  de la base de conocimiento, dilo y pide más detalles o escala el caso.
""".strip()


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
                f"Ya identificaste a {nombre_usuario}. Ahora pregúntale con qué problema "
                "necesita ayuda. Usa la herramienta buscar_en_base_conocimiento para buscar "
                "la solución antes de responder con procedimientos técnicos. Guíalo paso a paso. "
                "Si el usuario confirma que el problema quedó resuelto, llama a marcar_resuelto. "
                "Si después de intentar soluciones razonables el problema persiste, "
                "o el usuario lo pide, llama a escalar_a_humano."
            ),
        }],
        functions=[buscar_en_base_conocimiento, marcar_resuelto, escalar_a_humano],
    )


def crear_nodo_cierre(resuelto: bool) -> NodeConfig:
    mensaje = (
        "El problema quedó resuelto. Agradece a la persona y despídete brevemente."
        if resuelto else
        "El caso fue escalado a un agente humano. Avísale a la persona que alguien del "
        "equipo la contactará pronto, agradece su paciencia y despídete brevemente."
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

async def run_bot(transport, app_resources: AppResources):
    stt = ElevenLabsRealtimeSTTService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        params=ElevenLabsRealtimeSTTService.InputParams(
            language=Language.ES,
        ),
    )

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
    )

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice=os.environ.get("ELEVENLABS_VOICE_ID_COLOMBIA", "b2htR0pMe28pYwCY9gnP"),
            model="eleven_flash_v2_5",
            language=Language.ES,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(pipeline, app_resources=app_resources)

    flow_manager = FlowManager(
        llm=llm,
        context_aggregator=user_aggregator,
        worker=worker,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await flow_manager.initialize(crear_nodo_saludo())

    runner = WorkerRunner(handle_sigint=False)
    await runner.run(worker)


async def bot(runner_args: RunnerArguments):
    """Punto de entrada del bot — funciona igual en local y en Pipecat Cloud."""
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)

    app_resources = cargar_recursos()
    await run_bot(transport, app_resources)


if __name__ == "__main__":
    runner_main()
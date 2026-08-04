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

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat_flows import FlowManager, NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import CommitStrategy, ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.text.base_text_filter import BaseTextFilter
TRANSPORT = os.getenv("TRANSPORT", "local")  # daily

if TRANSPORT == "daily":
    from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Parche: Groq/Llama 3.3 a veces llama funciones sin parámetros (ej.
# marcar_resuelto) con arguments="null" en vez de "{}". pipecat convierte eso
# con json.loads() a None, y FlowsDirectFunctionWrapper.invoke() hace
# `self.function(flow_manager=flow_manager, **args)` sin validar — con
# args=None truena con "argument after ** must be a mapping, not NoneType".
# Se corrige aquí en vez de esperar a que la librería lo valide.
# ---------------------------------------------------------------------------

from pipecat_flows.types import FlowsDirectFunctionWrapper

_original_flows_direct_invoke = FlowsDirectFunctionWrapper.invoke


async def _flows_direct_invoke_tolerante_a_none(self, args, flow_manager):
    if args is None:
        args = {}
    return await _original_flows_direct_invoke(self, args, flow_manager)


FlowsDirectFunctionWrapper.invoke = _flows_direct_invoke_tolerante_a_none

# ---------------------------------------------------------------------------
# Config de RAG
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME = "soporte-ofimatico"
# Umbral heredado de cuando el embedding era MiniLM local; con multilingual-e5-large
# (embedding integrado de Pinecone) el score ya no es directamente comparable —
# recalibrar si el RAG rechaza matches que a simple oído sí son relevantes.
UMBRAL_CONFIANZA_RAG = 0.55

# stop_secs sube del default de Pipecat (0.2s) a 0.7s: con 0.2s, Silero declaraba el
# turno del usuario terminado más rápido de lo que ElevenLabs (vad_silence_threshold_secs
# más abajo) tardaba en comprometer la transcripción — el LLM recibía el turno vacío o
# a medias, y el texto real quedaba huérfano hasta el siguiente ciclo de habla. Un solo
# VAD (este) manda ahora sobre el corte de turno; ver commit_strategy=MANUAL en el STT.
VAD_PARAMS = VADParams(stop_secs=0.7)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS),
    ),
}


_recursos_rag_cache = None


def cargar_recursos_rag():
    """Conecta a Pinecone una sola vez por proceso (no por sesión/llamada) y
    reutiliza esa misma instancia en las siguientes.

    El índice usa embedding integrado de Pinecone (modelo multilingual-e5-large):
    Pinecone genera el vector tanto al ingestar (ingesta.py) como al buscar
    (buscar_en_base_conocimiento), así que el bot nunca calcula ni pide un
    embedding por su cuenta."""
    global _recursos_rag_cache
    if _recursos_rag_cache is None:
        logger.info("Conectando a Pinecone...")
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        indice = pc.Index(PINECONE_INDEX_NAME)
        _recursos_rag_cache = {"pinecone_index": indice}
    return _recursos_rag_cache


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


NOMBRES_PLACEHOLDER = {"nombre del usuario", "nombre de usuario", "usuario", "nombre", "john doe", "n/a", "desconocido"}


@tool_options(cancel_on_interruption=False)
async def identificar_usuario(flow_manager: FlowManager, nombre: str):
    """Registra el nombre de la persona que llama y avanza a la etapa de diagnóstico.

    Args:
        nombre: Nombre completo o como se identificó la persona que llama.
    """
    # Guardrail: a veces el LLM llama esta función con el placeholder literal
    # de la descripción del parámetro (p.ej. "nombre del usuario") en vez de
    # esperar a que la persona diga su nombre real. Si pasa, no avanzamos de
    # nodo y se lo decimos explícitamente al LLM para que vuelva a preguntar.
    if not nombre.strip() or nombre.strip().lower() in NOMBRES_PLACEHOLDER:
        logger.warning(f"identificar_usuario recibió un valor inválido: {nombre!r}")
        return {
            "identificado": False,
            "instruccion": "Ese no es un nombre real. Pide explícitamente el nombre y espera la respuesta antes de volver a llamar esta función.",
        }, None

    # Aquí iría la consulta real a tu base de datos de usuarios (por HTTP).
    # Se deja como stub porque esa base todavía no está construida.
    logger.info(f"Usuario identificado: {nombre}")
    flow_manager.state["nombre_usuario"] = nombre # Guardó el nombre en el estado compartido del flow:
    resultado = {"identificado": True, "nombre": nombre}
    return resultado, crear_nodo_diagnostico(nombre) # aqui avanza al siguiente nodo


async def buscar_en_base_conocimiento(flow_manager: FlowManager, consulta: str):
    """Busca en la base de conocimiento de soporte ofimático (Excel, Teams, SharePoint, cuentas Microsoft).
 
    Úsala cada vez que necesites un procedimiento de solución concreto,
    en vez de inventarlo de memoria.
 
    Args:
        consulta: La descripción del problema en las palabras del usuario, tal cual las dijo.
    """
    recursos = flow_manager.state["recursos_rag"]

    respuesta = await asyncio.to_thread(
        recursos["pinecone_index"].search,
        namespace="kb",
        top_k=3,
        inputs={"text": consulta},
        fields=["content", "producto", "url"],
    )

    hits = respuesta.result.hits
    if not hits or hits[0].score < UMBRAL_CONFIANZA_RAG:
        # Guardrail: si no hay match confiable, se lo decimos explícitamente
        # al LLM para que no invente una solución.
        # IMPORTANTE: Pipecat Flows siempre espera una tupla (resultado, next_node),
        # incluso para funciones que no cambian de nodo — en ese caso next_node = None.
        return {
            "encontrado": False,
            "instruccion": (
                "No se encontró información confiable. No inventes una solución. "
                "Pide más detalles o, si ya intentaste 2 veces sin éxito, ofrece escalar."
            ),
        }, None

    mejor = hits[0].fields
    return {
        "encontrado": True,
        "contenido": mejor["content"],
        "producto": mejor.get("producto"),
        "url_referencia": mejor.get("url"),
    }, None


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
                "Saluda brevemente y pide el nombre del usuario.\n\n"

                "IMPORTANTE:\n"
                "- Espera a que el usuario responda.\n"
                "- No inventes un nombre.\n"
                "- No llames identificar_usuario hasta que el usuario haya dicho explícitamente su nombre.\n"
                "- Tu único objetivo en este turno es saludar y pedir el nombre."
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
# Filtro de TTS: a veces Llama 3.3 (vía Groq) narra el tool call como texto
# plano en vez de (o además de) emitirlo como tool_call estructurado, p.ej.
# "<function=identificar_usuario>{...}</function>". El prompt ya pide no
# hacerlo, pero no es suficiente, así que lo cortamos antes de que llegue a
# TTS. Es stateful porque el texto llega en fragmentos y la etiqueta puede
# quedar partida entre un fragmento y el siguiente.
# ---------------------------------------------------------------------------

class FunctionCallLeakFilter(BaseTextFilter):
    """Elimina restos de sintaxis de tool calls (`<function=...>...</function>`)
    que el LLM a veces filtra como texto normal en vez de como tool_call real."""

    _START = "<function"
    _END = "</function>"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._suppressing = False

    async def filter(self, text: str) -> str:
        result = []
        remaining = text
        while remaining:
            if self._suppressing:
                end_idx = remaining.find(self._END)
                if end_idx == -1:
                    remaining = ""  # seguimos suprimiendo en el próximo fragmento
                    break
                remaining = remaining[end_idx + len(self._END) :]
                self._suppressing = False
                continue

            start_idx = remaining.lower().find(self._START)
            if start_idx == -1:
                result.append(remaining)
                remaining = ""
                break

            result.append(remaining[:start_idx])
            remaining = remaining[start_idx:]
            self._suppressing = True

        return "".join(result)

    async def handle_interruption(self):
        self._suppressing = False

    async def reset_interruption(self):
        pass


# ---------------------------------------------------------------------------
# Filtro de alucinaciones del STT: en audio corto/ambiguo (ruido, respiración,
# silencio con VAD sensible), Scribe a veces "escucha" una muletilla típica de
# call center en vez de devolver texto vacío. Se descartan por igualdad exacta
# (no por substring) para no tocar frases reales que solo contengan estas
# palabras (ej. "¿qué debo hacer?" sí pasa; "¿qué?" a secas no).
# ---------------------------------------------------------------------------

MULETILLAS_FANTASMA = {"hola", "aló", "alo", "qué", "que", "eh", "ah"}


class AsrHallucinationFilter(FrameProcessor):
    """Descarta transcripciones (finales o parciales) que sean, en su totalidad,
    una de las muletillas fantasma que el STT alucina sobre audio ambiguo."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            normalizado = frame.text.strip(" ¿?¡!.,").lower()
            if normalizado in MULETILLAS_FANTASMA:
                logger.debug(f"Descartada posible alucinación del STT: {frame.text!r}")
                return

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Pipeline y entry point del bot
# ---------------------------------------------------------------------------

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        # MANUAL: el VAD de Pipecat (Silero, ver VAD_PARAMS) es la única fuente de
        # verdad para el corte de turno y le manda el commit a ElevenLabs — evita la
        # carrera que había entre el VAD interno de ElevenLabs y el de Pipecat, donde
        # el turno se cerraba antes de que la transcripción llegara.
        commit_strategy=CommitStrategy.MANUAL,
        settings=ElevenLabsRealtimeSTTService.Settings(
            model="scribe_v2_realtime",
            language="es",
        ),
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY", ""),
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # más determinismo: reduce (no elimina) que el modelo narre el tool call como texto en vez de emitirlo estructurado
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        text_filters=[FunctionCallLeakFilter()],
        settings=ElevenLabsTTSService.Settings(
            voice="6Gr4AVmTax1pMJO0lHRK",  # Chile: 6Gr4AVmTax1pMJO0lHRK - Colombia : b2htR0pMe28pYwCY9gnP
            model="eleven_flash_v2_5",
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS),
            filter_incomplete_user_turns=False,  # Mega Prompt para que el LLM no se quede esperando a que el usuario termine de hablar, sino que procese lo que haya dicho hasta ahora.
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        AsrHallucinationFilter(),
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

    # Recursos de RAG disponibles para cualquier función de Flows vía flow_manager.state.
    # Solo conecta a Pinecone (embeddings integrados, ver cargar_recursos_rag) — no
    # carga ningún modelo pesado en memoria.
    flow_manager.state["recursos_rag"] = cargar_recursos_rag()

    conversacion_iniciada = False

    async def iniciar_conversacion():
        nonlocal conversacion_iniciada
        if conversacion_iniciada:
            return
        conversacion_iniciada = True
        logger.info("Cliente conectado")
        # Espera a que el track de audio del participante termine de negociarse
        # en WebRTC antes de saludar. Sin esto, el saludo (LLM+TTS, ambos muy
        # rápidos) se genera y reproduce apenas llega el evento de conexión, que
        # dispara antes de que el navegador esté realmente suscrito al audio del
        # bot — el usuario se pierde el saludo y percibe silencio hasta que él
        # mismo habla primero.
        await asyncio.sleep(1.5)
        await flow_manager.initialize(crear_nodo_saludo())

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await iniciar_conversacion()

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # on_client_connected solo dispara para participantes que se unen
        # DESPUÉS del bot. Si el bot tuvo un cold start lento y el usuario ya
        # estaba en la sala cuando el bot terminó de unirse, ese evento nunca
        # llega — lo detectamos aquí como respaldo revisando quién ya está.
        local_id = data.get("participants", {}).get("local", {}).get("id")
        if any(pid != local_id for pid in transport.participants()):
            logger.info("Ya había un participante en la sala al unirse el bot")
            await iniciar_conversacion()

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
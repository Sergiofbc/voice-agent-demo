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
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService

from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService, CommitStrategy
from pipecat.transports.base_transport import TransportParams, BaseTransport
TRANSPORT = os.getenv("TRANSPORT", "local")

if TRANSPORT == "daily":
    from pipecat.transports.daily.transport import DailyParams

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt del agente: soporte técnico ofimático (Office/Windows/impresoras/etc.)
# Conversacional: diagnostica preguntando antes de proponer soluciones.
# ---------------------------------------------------------------------------
HELPDESK_SYSTEM_PROMPT = """Eres Sofía, agente de soporte técnico de la mesa de ayuda de sistemas ofimáticos de la empresa (Word, Excel, Outlook, impresoras, VPN, correo, redes compartidas, etc.).

OBJETIVO:
Ayudar al usuario a resolver su problema técnico por teléfono, de forma conversacional, haciendo preguntas para diagnosticar antes de dar una solución. No asumas el problema: pregunta primero.

FLUJO DE LA LLAMADA:
1. Saluda brevemente y pregunta el nombre del usuario y qué problema está presentando.
2. Haz preguntas de diagnóstico UNA A LA VEZ (no varias juntas): qué programa o equipo, qué mensaje de error ve exactamente, cuándo empezó, si ya reinició el equipo, si le pasa a otros compañeros también, si hizo algún cambio reciente (actualización, instalación, cambio de contraseña, etc.).
3. Con la información recolectada, propone una solución paso a paso, UN PASO A LA VEZ, y espera confirmación del usuario antes de continuar al siguiente paso.
4. Si el paso resuelve el problema, confirma que quedó solucionado y pregunta si necesita algo más.
5. Si el paso no resuelve el problema, prueba la siguiente alternativa razonable. Si tras 2-3 intentos no se resuelve, ofrece escalar el caso a un técnico humano, indicando que se creará un ticket de soporte.
6. Cierra la llamada con un resumen breve de lo realizado (o del ticket creado) y despídete cordialmente.

REGLAS:
- Responde en español neutro, con tono cálido, profesional y paciente.
- Turnos CORTOS: 1-3 frases por intervención. Es una llamada telefónica en vivo, no un monólogo. Espera la respuesta del usuario antes de seguir.
- No des por hecho detalles técnicos que el usuario no ha mencionado (marca de equipo, sistema operativo, versión de Office, etc.) — pregúntalos si son relevantes para el diagnóstico.
- No prometas tiempos de resolución que no puedas confirmar.
- Si el usuario está frustrado o molesto, reconoce la situación con empatía antes de continuar con el diagnóstico.
- Si el problema es urgente (por ejemplo, no puede facturar o no puede trabajar en absoluto), prioriza ofrecer la escalación a un técnico humano más rápido.
- Si detectas que el problema no es de soporte ofimático (por ejemplo, un tema de nómina o de recursos humanos), indícalo con amabilidad y ofrece derivar la llamada al área correspondiente.
- Nunca inventes procedimientos técnicos que no tengan sentido; si no estás segura de un paso, ofrece escalar en vez de arriesgar una solución incorrecta.
"""


def get_transport_params():
    """Parámetros de transporte por tipo de conexión.

    Se usan lambdas para no instanciar nada (p.ej. SileroVADAnalyzer)
    hasta que se sepa qué transporte se va a usar realmente.
    """
    return {
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = ElevenLabsRealtimeSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        params=ElevenLabsRealtimeSTTService.InputParams(
            language_code="es",
            commit_strategy=CommitStrategy.VAD,
            vad_silence_threshold_secs=0.7,
        ),
    )  

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY", ""),
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        settings=ElevenLabsTTSService.Settings(
            voice="6Gr4AVmTax1pMJO0lHRK",  # Chile: 6Gr4AVmTax1pMJO0lHRK - Colombia : b2htR0pMe28pYwCY9gnP
            model="eleven_flash_v2_5",
        ),
    )

    # guarda la conversacion (COMPLETO)
    context = LLMContext()
    context.add_message({"role": "system", "content": HELPDESK_SYSTEM_PROMPT})

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(),  # sin vad_analyzer aquí
    )

    pipeline = Pipeline(
        [
            transport.input(),  # origen de los datos (WebRTC o Daily) - recibe audio
            stt,
            user_aggregator,  # guardar mensaje usuario
            llm,
            tts,
            transport.output(),  # envia audio al participante
            assistant_aggregator,
        ]
    )

    # ejecuta el pipeline en un worker asincrono, a la escucha de nuevos eventos
    # del transporte (conexiones, desconexiones, nuevos audios)
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
        context.add_message(
            {
                "role": "developer",
                "content": "Saluda al usuario y pregúntale su nombre y qué problema técnico está presentando.",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.run(worker)


# punto de entrada del bot — funciona igual en local (webrtc) y en Daily/Pipecat Cloud
async def bot(runner_args: RunnerArguments):
    logger.info(f"bot() called with: {type(runner_args)}")
    transport = await create_transport(runner_args, get_transport_params())
    await run_bot(transport, runner_args)


# main de pipecat runner
if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
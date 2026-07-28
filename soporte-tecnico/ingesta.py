"""
Script de ingesta para la base de conocimiento del bot de soporte ofimático.

Flujo:
    JSON de catálogo (producto -> {pregunta: url})
      -> descarga la URL
      -> extrae solo el contenido útil del artículo (descarta menú/footer/encuesta)
      -> trocea el contenido en chunks
      -> genera embeddings localmente (open source, sin costo de API)
      -> sube todo a Pinecone

Requisitos (instalar una vez):
    pip install httpx trafilatura sentence-transformers "pinecone[grpc]" tqdm

Antes de correr:
    - Crea una cuenta gratis en pinecone.io y pon tu API key abajo (o en variable de entorno).
    - Ajusta CATALOGO_PATH a la ruta de tu archivo JSON.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import httpx
import trafilatura
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError(
        "No se encontró PINECONE_API_KEY"
    )

PINECONE_INDEX_NAME = "soporte-ofimatico"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"  # región del free tier de Pinecone

# Modelo de embeddings open source, local, gratis.
# Soporta español, portugués e inglés (y ~50 idiomas más).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE_PALABRAS = 350
CHUNK_OVERLAP_PALABRAS = 50

CATALOGO_PATH = "KB.json"

# Frases de ruido que a veces trafilatura no logra descartar del todo
# (encuestas de feedback, banners de "fue útil esta info", etc.)
FRASES_DE_CORTE = [
    "Was this information helpful?",
    "¿Te ha sido útil esta información?",
    "What affected your experience?",
    "Submit feedback",
    "Thank you for your feedback!",
]


# ---------------------------------------------------------------------------
# 1. Descarga y limpieza
# ---------------------------------------------------------------------------

def descargar_y_limpiar(url: str) -> str:
    """Descarga el HTML y extrae solo el contenido principal del artículo."""
    respuesta = httpx.get(
        url,
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; IngestaBot/1.0)"},
    )
    respuesta.raise_for_status()

    texto = trafilatura.extract(
        respuesta.text,
        include_links=False,
        include_images=False,
        include_tables=True,
        favor_recall=True,  # prioriza no perder contenido útil sobre limpieza perfecta
    )

    if not texto:
        raise ValueError(f"trafilatura no pudo extraer contenido de: {url}")

    return _limpiar_ruido_residual(texto)


def _limpiar_ruido_residual(texto: str) -> str:
    """Corta cualquier cola de encuesta/feedback que haya sobrevivido a trafilatura
    y colapsa espacios/saltos de línea repetidos."""
    for frase in FRASES_DE_CORTE:
        if frase in texto:
            texto = texto.split(frase)[0]

    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

# Umbral: si una sección de encabezado supera esto, además se subdivide por palabras
MAX_PALABRAS_POR_SECCION = 400


def _trocear_por_palabras(texto: str, tam_palabras: int = CHUNK_SIZE_PALABRAS,
                           solape: int = CHUNK_OVERLAP_PALABRAS) -> list[str]:
    palabras = texto.split()
    if not palabras:
        return []
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + tam_palabras
        chunk = " ".join(palabras[inicio:fin])
        if chunk.strip():
            chunks.append(chunk)
        if fin >= len(palabras):
            break
        inicio += tam_palabras - solape
    return chunks


def trocear(texto: str) -> list[str]:
    """
    Trocea respetando la estructura del artículo: cada sección '## Encabezado'
    se convierte en un chunk propio (con su encabezado incluido como contexto),
    para no partir un procedimiento de solución a la mitad de sus pasos.

    Si una sección es muy larga, se subdivide por palabras como fallback.
    Los artículos de troubleshooting de Microsoft usan '##' para cada causa/solución,
    así que esto suele producir chunks ya alineados 1 a 1 con "una solución = un chunk".
    """
    # Separa por encabezados de nivel 2 (## ...), conservando el encabezado en el chunk
    secciones = re.split(r"(?=^## )", texto, flags=re.MULTILINE)

    chunks = []
    for seccion in secciones:
        seccion = seccion.strip()
        if not seccion:
            continue

        if len(seccion.split()) <= MAX_PALABRAS_POR_SECCION:
            chunks.append(seccion)
        else:
            # Sección muy larga: conserva el título y subdivide el resto por palabras
            lineas = seccion.split("\n", 1)
            titulo = lineas[0]
            cuerpo = lineas[1] if len(lineas) > 1 else ""
            for sub in _trocear_por_palabras(cuerpo):
                chunks.append(f"{titulo}\n{sub}")

    # Si el artículo no tenía encabezados '##' en absoluto, cae al corte por palabras
    if not chunks:
        chunks = _trocear_por_palabras(texto)

    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings (open source, corre localmente, sin costo por llamada)
# ---------------------------------------------------------------------------

print(f"Cargando modelo de embeddings ({EMBEDDING_MODEL_NAME})...")
modelo_embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME)
DIMENSION = modelo_embeddings.get_embedding_dimension()


def embed(texto: str) -> list[float]:
    vector = modelo_embeddings.encode(texto, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# 4. Pinecone
# ---------------------------------------------------------------------------

def obtener_o_crear_indice():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    nombres_existentes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in nombres_existentes:
        print(f"Creando índice '{PINECONE_INDEX_NAME}' (dim={DIMENSION})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def slugify(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def procesar_catalogo(catalogo: dict, indice) -> None:
    """
    Espera un catálogo con forma:
    {
      "Excel": {"No puedo editar archivo excel": "https://..."},
      "Teams": {"...": "..."},
    }
    """
    lote = []
    LOTE_MAX = 100  # tamaño de batch para el upsert a Pinecone

    for producto, casos in catalogo.items():
        for pregunta, url in tqdm(casos.items(), desc=f"Procesando {producto}"):
            doc_id = f"{slugify(producto)}__{slugify(pregunta)}"

            try:
                texto_limpio = descargar_y_limpiar(url)
            except Exception as e:
                print(f"  [ERROR] {url} -> {e}")
                continue

            chunks = trocear(texto_limpio)
            if not chunks:
                print(f"  [AVISO] Sin contenido útil en: {url}")
                continue

            # El primer chunk suele contener el resumen/causa principal;
            # es lo que se copia como 'content' del vector de título.
            contenido_principal = chunks[0]

            # Vector de la pregunta en lenguaje natural: el "gancho" de búsqueda.
            lote.append({
                "id": f"{doc_id}__titulo",
                "values": embed(pregunta),
                "metadata": {
                    "doc_id": doc_id,
                    "producto": producto,
                    "tipo": "titulo",
                    "url": url,
                    "content": contenido_principal,
                },
            })

            # Vectores del contenido real, trozo por trozo.
            for i, chunk in enumerate(chunks):
                lote.append({
                    "id": f"{doc_id}__chunk_{i}",
                    "values": embed(chunk),
                    "metadata": {
                        "doc_id": doc_id,
                        "producto": producto,
                        "tipo": "contenido",
                        "url": url,
                        "content": chunk,
                    },
                })

            if len(lote) >= LOTE_MAX:
                indice.upsert(vectors=lote)
                lote = []

    if lote:
        indice.upsert(vectors=lote)


if __name__ == "__main__":
    catalogo = json.loads(Path(CATALOGO_PATH).read_text(encoding="utf-8"))
    indice = obtener_o_crear_indice()
    procesar_catalogo(catalogo, indice)
    print("Ingesta completa.")
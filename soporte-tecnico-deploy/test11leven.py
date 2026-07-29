import os, requests
from dotenv import load_dotenv
load_dotenv()

resp = requests.post(
    "https://api.elevenlabs.io/v1/text-to-speech/6Gr4AVmTax1pMJO0lHRK",
    headers={
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
        "Content-Type": "application/json",
    },
    json={"text": "Hola, esto es una prueba.", "model_id": "eleven_multilingual_v2"},
)
print("Status:", resp.status_code)
print("Headers:", resp.headers.get("content-type"))
print("Bytes recibidos:", len(resp.content))

if resp.status_code == 200:
    with open("test.mp3", "wb") as f:
        f.write(resp.content)
    print("Guardado test.mp3 — ábrelo y escúchalo")
else:
    print("Body:", resp.text)
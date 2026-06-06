"""
yolo_server.py — YOLOv8n + Moondream2 (Ollama) · 100% local · sin costo
========================================================================
Puerto: 8000

Requisitos:
    pip install fastapi uvicorn ultralytics pillow numpy

Ollama (instalar desde ollama.com) y luego:
    ollama pull moondream

Arranque:
    python yolo_server.py
"""

import base64
import io
import json
import colorsys
import urllib.request
import urllib.error
from collections import Counter
from typing import Optional

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Modelo YOLOv8n (se carga una sola vez al arrancar) ───────────────────────
from ultralytics import YOLO
print("Cargando YOLOv8n…")
model = YOLO("yolov8n.pt")
print("YOLOv8n listo.")

# ── Configuración Ollama ──────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/chat"
MOON_MODEL  = "moondream"

# ── Traducción COCO → español ─────────────────────────────────────────────────
COCO_ES = {
    "person":"persona","bicycle":"bicicleta","car":"automóvil","motorcycle":"motocicleta",
    "airplane":"avión","bus":"autobús","train":"tren","truck":"camión","boat":"bote",
    "traffic light":"semáforo","fire hydrant":"hidrante","stop sign":"señal de pare",
    "parking meter":"parquímetro","bench":"banco","bird":"pájaro","cat":"gato",
    "dog":"perro","horse":"caballo","sheep":"oveja","cow":"vaca","elephant":"elefante",
    "bear":"oso","zebra":"cebra","giraffe":"jirafa","backpack":"mochila","umbrella":"paraguas",
    "handbag":"bolso","tie":"corbata","suitcase":"maleta","frisbee":"frisbee","skis":"esquís",
    "snowboard":"snowboard","sports ball":"balón","kite":"cometa","baseball bat":"bate",
    "baseball glove":"guante béisbol","skateboard":"patineta","surfboard":"tabla surf",
    "tennis racket":"raqueta tenis","bottle":"botella","wine glass":"copa de vino",
    "cup":"taza","fork":"tenedor","knife":"cuchillo","spoon":"cuchara","bowl":"tazón",
    "banana":"plátano","apple":"manzana","sandwich":"sándwich","orange":"naranja",
    "broccoli":"brócoli","carrot":"zanahoria","hot dog":"perro caliente","pizza":"pizza",
    "donut":"dona","cake":"pastel","chair":"silla","couch":"sofá","potted plant":"planta",
    "bed":"cama","dining table":"mesa comedor","toilet":"inodoro","tv":"televisor",
    "laptop":"portátil","mouse":"ratón","remote":"control remoto","keyboard":"teclado",
    "cell phone":"teléfono celular","microwave":"microondas","oven":"horno",
    "toaster":"tostadora","sink":"lavabo","refrigerator":"refrigerador","book":"libro",
    "clock":"reloj","vase":"jarrón","scissors":"tijeras","teddy bear":"oso de peluche",
    "hair drier":"secador","toothbrush":"cepillo dental"
}

app = FastAPI(title="YOLOv8n + Moondream2 Local API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Ollama: verificar disponibilidad ─────────────────────────────────────────
def ollama_disponible() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ── Ollama: llamar a moondream con imagen ────────────────────────────────────
def describir_con_moondream(img: Image.Image) -> dict:
    """
    Envía la imagen a moondream2 vía Ollama y devuelve un dict semántico compacto.
    Hace 3 preguntas separadas para obtener descripción, escena y prompt.
    """

    # Escalar imagen a máx 512px para ahorrar RAM y ser más rápido
    thumb = img.copy()
    thumb.thumbnail((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    def preguntar(prompt_text: str) -> str:
        payload = json.dumps({
            "model": MOON_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt_text,
                "images": [img_b64]
            }],
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode())
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"[error: {e}]"

    # Pregunta 1: descripción general
    descripcion = preguntar(
        "Describe this image in 2-3 sentences. Be specific about what you see: "
        "people, objects, setting, colors, mood. Answer in Spanish."
    )

    # Pregunta 2: escena y ambiente
    escena_raw = preguntar(
        "In one sentence each, answer: "
        "1) Is this indoors or outdoors? "
        "2) What is the approximate time of day? "
        "3) What is the general mood or atmosphere? "
        "Answer in Spanish, keep it very brief."
    )

    # Pregunta 3: prompt de recreación
    prompt_recreacion = preguntar(
        "Write a detailed prompt in English to recreate this image with an AI image generator. "
        "Include: subject, setting, lighting, colors, style, composition. Max 2 sentences."
    )

    return {
        "descripcion_moondream": descripcion,
        "escena_moondream": escena_raw,
        "prompt_recreacion": prompt_recreacion,
        "motor_semantico": "moondream2 (ollama local)"
    }


# ── Análisis técnico de imagen ────────────────────────────────────────────────
def get_brightness(arr: np.ndarray) -> float:
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    return round(float(gray.mean()) / 255, 4)

def get_saturation(arr: np.ndarray) -> float:
    sats = []
    sample = arr[::6, ::6]
    for row in sample:
        for px in row:
            r, g, b = px[0]/255, px[1]/255, px[2]/255
            _, s, _ = colorsys.rgb_to_hsv(r, g, b)
            sats.append(s)
    return round(float(np.mean(sats)), 4) if sats else 0.0

def extract_palette(arr: np.ndarray, n: int = 6) -> list:
    pixels = arr[::8, ::8].reshape(-1, 3)
    quantized = (pixels // 32) * 32
    counts = Counter(map(tuple, quantized))
    top = counts.most_common(n)
    palette = []
    for (r, g, b), _ in top:
        hex_c = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
        palette.append({"hex": hex_c, "rgb": [int(r), int(g), int(b)]})
    return palette

def color_names(palette: list) -> list:
    NAMES = {
        (0,30):"rojo",(30,60):"naranja",(60,90):"amarillo",
        (90,150):"verde",(150,180):"cian",(180,240):"azul",
        (240,270):"índigo",(270,330):"violeta",(330,360):"rosa"
    }
    names = []
    for e in palette[:5]:
        r, g, b = [x/255 for x in e["rgb"]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360
        if s < 0.15:
            name = "blanco" if v > 0.8 else ("gris" if v > 0.3 else "negro")
        else:
            name = "rojo"
            for (lo, hi), n in NAMES.items():
                if lo <= h_deg < hi:
                    name = n
                    break
        if name not in names:
            names.append(name)
    return names

def build_thumbnail_b64(img: Image.Image, max_side: int = 128) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def build_pixel_matrix(img: Image.Image, step: int = 8) -> dict:
    small = img.resize(
        (max(1, img.width // step), max(1, img.height // step)),
        Image.LANCZOS
    )
    arr = np.array(small.convert("RGB"))
    rows, cols, _ = arr.shape
    matrix = [[[int(px[0]), int(px[1]), int(px[2])] for px in row] for row in arr]
    return {"filas": rows, "columnas": cols, "paso_muestreo": step, "datos": matrix}


# ── Análisis principal ────────────────────────────────────────────────────────
def analyze_image(img: Image.Image) -> dict:
    arr = np.array(img.convert("RGB"))

    # 1. YOLO
    results = model(img, verbose=False)[0]
    boxes = results.boxes
    detections = []
    objects_en = []

    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id  = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf    = float(box.conf[0])
            xyxy    = box.xyxy[0].tolist()
            objects_en.append(cls_name)
            detections.append({
                "clase":    cls_name,
                "clase_es": COCO_ES.get(cls_name, cls_name),
                "confianza": round(conf, 3),
                "bbox": {
                    "x1": round(xyxy[0]), "y1": round(xyxy[1]),
                    "x2": round(xyxy[2]), "y2": round(xyxy[3])
                }
            })

    objects_es = list(dict.fromkeys([COCO_ES.get(o, o) for o in objects_en]))
    personas   = [d for d in detections if d["clase"] == "person"]

    # 2. Técnico
    brightness  = get_brightness(arr)
    saturation  = get_saturation(arr)
    palette     = extract_palette(arr)
    colors      = color_names(palette)
    thumbnail   = build_thumbnail_b64(img)
    matrix      = build_pixel_matrix(img)

    # 3. Moondream2 (semántica compacta) — si Ollama está disponible
    moon_data = {}
    ollama_ok = ollama_disponible()
    if ollama_ok:
        print("Consultando moondream2…")
        moon_data = describir_con_moondream(img)
        print("Moondream listo.")
    else:
        moon_data = {
            "descripcion_moondream": "Ollama no disponible. Instala Ollama y ejecuta: ollama pull moondream",
            "escena_moondream": "no disponible",
            "prompt_recreacion": "no disponible",
            "motor_semantico": "sin moondream"
        }

    # Sujeto principal
    sujeto = detections[0]["clase_es"] if detections else "escena sin objetos COCO detectados"

    # Descripción de respaldo si moondream no está
    desc_local = ""
    n_p = len(personas)
    otros = [o for o in objects_es if o != "persona"]
    if n_p == 1: desc_local += "Una persona en la imagen. "
    elif n_p > 1: desc_local += f"{n_p} personas en la imagen. "
    if otros: desc_local += f"Objetos: {', '.join(otros[:5])}. "
    desc_local += f"Brillo {brightness:.0%}, colores: {', '.join(colors[:3])}."

    descripcion_final = moon_data.get("descripcion_moondream") or desc_local

    # Keywords
    keywords = list(dict.fromkeys(objects_es[:4] + colors[:2]))[:8]

    return {
        # ── Semántica compacta (lo que cabe en media página) ──────────────
        "sujeto":               sujeto,
        "descripcion_semantica": descripcion_final,
        "escena_descripcion":   moon_data.get("escena_moondream", ""),
        "objetos_detectados":   objects_es,
        "personas_count":       len(personas),
        "colores_dominantes":   colors,
        "palabras_clave":       keywords,
        "prompt_recreacion":    moon_data.get("prompt_recreacion", ""),
        "motor_semantico":      moon_data.get("motor_semantico", "yolo-solo"),

        # ── Detecciones YOLO con coordenadas ──────────────────────────────
        "detecciones_yolo": detections,

        # ── Técnico ───────────────────────────────────────────────────────
        "estructura": {
            "ancho":    img.width,
            "alto":     img.height,
            "brillo":   brightness,
            "saturacion": saturation,
            "ollama_activo": ollama_ok
        },

        # ── Reconstrucción ────────────────────────────────────────────────
        "reconstruccion": {
            "thumbnail_base64": thumbnail,
            "paleta_completa":  palette,
            "matriz_pixeles":   matrix
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    ollama_ok = ollama_disponible()
    return {
        "status": "ok",
        "motores": {
            "yolov8n": "activo",
            "moondream2": "activo" if ollama_ok else "inactivo — ejecuta: ollama pull moondream"
        }
    }

@app.post("/api/seresinmortales")
def ping():
    return {"message": "Endpoint activo.", "motores": ["YOLOv8n", "moondream2 (Ollama)"]}

@app.post("/extraer-json")
async def extraer_json(
    imagen:   Optional[UploadFile] = File(None),
    base64:   Optional[str]        = Form(None),
    mimeType: Optional[str]        = Form("image/jpeg")
):
    try:
        if imagen is not None:
            raw = await imagen.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif base64 is not None:
            raw_b64 = base64
            for prefix in ["data:image/jpeg;base64,","data:image/png;base64,","data:image/webp;base64,"]:
                raw_b64 = raw_b64.replace(prefix, "")
            raw = __import__("base64").b64decode(raw_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail='Envía imagen como multipart o base64.')

        result = analyze_image(img)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

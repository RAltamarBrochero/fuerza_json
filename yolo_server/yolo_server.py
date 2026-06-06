"""
yolo_server.py — Servidor local YOLOv8n para Mi_Api_4
======================================================
Corre en http://localhost:8000
Recibe una imagen (multipart o base64) y devuelve un JSON completo
para describir Y reconstruir la imagen, usando SOLO recursos locales.

Requisitos:
    pip install fastapi uvicorn ultralytics pillow numpy

Arranque:
    python yolo_server.py
"""

import base64
import io
import json
import math
import colorsys
from collections import Counter
from typing import Optional

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Carga del modelo (una sola vez al arrancar) ───────────────────────────────
from ultralytics import YOLO
print("Cargando modelo YOLOv8n …")
model = YOLO("yolov8n.pt")
print("Modelo listo.")

# ── Mapas de traducción COCO → español ───────────────────────────────────────
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
    "cell phone":"teléfono celular","microwave":"microondas","oven":"horno","toaster":"tostadora",
    "sink":"lavabo","refrigerator":"refrigerador","book":"libro","clock":"reloj","vase":"jarrón",
    "scissors":"tijeras","teddy bear":"oso de peluche","hair drier":"secador","toothbrush":"cepillo dental"
}

app = FastAPI(title="YOLOv8n Local API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Utilidades de análisis de imagen ─────────────────────────────────────────

def get_dimensions(img: Image.Image):
    return {"ancho": img.width, "alto": img.height}


def image_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def get_brightness(arr: np.ndarray) -> float:
    """Brillo medio normalizado 0-1."""
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return round(float(gray.mean()) / 255, 4)


def get_saturation(arr: np.ndarray) -> float:
    """Saturación media HSV normalizada 0-1."""
    sats = []
    sample = arr[::4, ::4]  # submuestreo para velocidad
    for row in sample:
        for px in row:
            r, g, b = px[0] / 255, px[1] / 255, px[2] / 255
            _, s, _ = colorsys.rgb_to_hsv(r, g, b)
            sats.append(s)
    return round(float(np.mean(sats)), 4) if sats else 0.0


def extract_palette(arr: np.ndarray, n_colors: int = 6) -> list:
    """Paleta de colores dominantes via cuantización simple."""
    pixels = arr[::8, ::8].reshape(-1, 3)
    # Cuantizar a 32 niveles por canal
    quantized = (pixels // 32) * 32
    counts = Counter(map(tuple, quantized))
    top = counts.most_common(n_colors)
    palette = []
    for (r, g, b), count in top:
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
        palette.append({"hex": hex_color, "rgb": [int(r), int(g), int(b)]})
    return palette


def dominant_color_names(palette: list) -> list:
    """Nombre aproximado de color a partir de tono HSV."""
    COLOR_NAMES = {
        (0, 30): "rojo", (30, 60): "naranja", (60, 90): "amarillo",
        (90, 150): "verde", (150, 180): "cian", (180, 240): "azul",
        (240, 270): "índigo", (270, 330): "violeta", (330, 360): "rosa"
    }
    names = []
    for entry in palette[:5]:
        r, g, b = [x / 255 for x in entry["rgb"]]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360
        if s < 0.15:
            name = "blanco" if v > 0.8 else ("gris" if v > 0.3 else "negro")
        else:
            name = "rojo"
            for (lo, hi), n in COLOR_NAMES.items():
                if lo <= h_deg < hi:
                    name = n
                    break
        if name not in names:
            names.append(name)
    return names


def infer_image_type(objects: list, has_text: bool) -> str:
    if has_text:
        return "documento"
    if not objects:
        return "foto"
    return "foto"


def infer_scene(objects: list, brightness: float) -> dict:
    outdoor_cues = {"car", "truck", "bus", "bicycle", "motorcycle", "bird", "dog",
                    "cat", "traffic light", "stop sign", "bench", "person"}
    indoor_cues = {"chair", "couch", "bed", "dining table", "tv", "laptop", "book",
                   "refrigerator", "microwave", "oven", "sink", "toilet"}
    obj_set = set(objects)
    outdoor_score = len(obj_set & outdoor_cues)
    indoor_score = len(obj_set & indoor_cues)

    if outdoor_score > indoor_score:
        tipo = "exterior"
        lugar = "espacio urbano o exterior"
    elif indoor_score > 0:
        tipo = "interior"
        lugar = "ambiente interior"
    else:
        tipo = "no_determinado"
        lugar = "no determinado"

    momento = "dia" if brightness > 0.45 else ("noche" if brightness < 0.2 else "no_determinado")

    return {
        "tipo": tipo,
        "lugar_probable": lugar,
        "momento_dia": momento,
        "clima": "no_aplica",
        "ambiente": "Detectado automáticamente por análisis local"
    }


def infer_composition(width: int, height: int) -> dict:
    ratio = width / height if height else 1
    if ratio > 2.5:
        plano = "panoramico"
    elif ratio < 0.6:
        plano = "plano_detalle"
    else:
        plano = "plano_general"
    return {"plano": plano, "angulo": "frontal"}


def build_prompt_recreacion(objects_es: list, colors: list, scene: dict, brightness: float) -> str:
    objs = ", ".join(objects_es[:6]) if objects_es else "escena"
    cols = ", ".join(colors[:3]) if colors else "colores naturales"
    lighting = "bright daylight" if brightness > 0.6 else ("dark atmosphere" if brightness < 0.25 else "natural lighting")
    ambiente = scene.get("tipo", "exterior")
    return (f"Photorealistic scene with {objs}. {ambiente.capitalize()} environment. "
            f"Dominant colors: {cols}. {lighting}. "
            f"High detail, natural composition, professional photography style.")


def build_pixel_matrix(img: Image.Image, sample_step: int = 8) -> dict:
    """
    Matriz de píxeles submuestreada para reconstrucción aproximada.
    sample_step=8 → 1 píxel por cada 8 → imagen ~8 veces más pequeña.
    """
    small = img.resize(
        (max(1, img.width // sample_step), max(1, img.height // sample_step)),
        Image.LANCZOS
    )
    arr = np.array(small.convert("RGB"))
    rows, cols, _ = arr.shape
    matrix = []
    for row in arr:
        matrix.append([[int(px[0]), int(px[1]), int(px[2])] for px in row])
    return {
        "filas": rows,
        "columnas": cols,
        "paso_muestreo": sample_step,
        "datos": matrix
    }


def build_thumbnail_b64(img: Image.Image, max_side: int = 128) -> str:
    """Miniatura en base64 PNG para reconstrucción rápida."""
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Análisis principal ────────────────────────────────────────────────────────

def analyze_image(img: Image.Image) -> dict:
    arr = image_to_numpy(img)

    # 1. YOLO detection
    results = model(img, verbose=False)[0]
    boxes = results.boxes

    detected_objects_en = []
    detections_detail = []

    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detected_objects_en.append(cls_name)
            detections_detail.append({
                "clase": cls_name,
                "clase_es": COCO_ES.get(cls_name, cls_name),
                "confianza": round(conf, 3),
                "bbox": {
                    "x1": round(xyxy[0]), "y1": round(xyxy[1]),
                    "x2": round(xyxy[2]), "y2": round(xyxy[3])
                }
            })

    detected_objects_es = [COCO_ES.get(o, o) for o in detected_objects_en]
    detected_unique_es = list(dict.fromkeys(detected_objects_es))  # sin duplicados

    # Personas detectadas
    personas = []
    for det in detections_detail:
        if det["clase"] == "person":
            personas.append({
                "descripcion": "Persona detectada por YOLOv8n",
                "genero_aparente": "no_determinado",
                "edad_aproximada": "no_determinado",
                "expresion": "no_determinado",
                "ropa": "no_determinado",
                "accion": "presente en la escena",
                "bbox": det["bbox"],
                "confianza": det["confianza"]
            })

    # 2. Análisis de color
    brightness = get_brightness(arr)
    saturation = get_saturation(arr)
    palette = extract_palette(arr)
    color_names = dominant_color_names(palette)

    # 3. Inferencias de escena
    scene = infer_scene(detected_objects_en, brightness)
    composition = infer_composition(img.width, img.height)
    img_type = infer_image_type(detected_objects_en, False)

    # Sujeto principal
    if detections_detail:
        sujeto_principal = detections_detail[0]["clase_es"]
    elif detected_unique_es:
        sujeto_principal = detected_unique_es[0]
    else:
        sujeto_principal = "escena sin objetos reconocidos por YOLO"

    # Descripción semántica generada localmente
    n_persons = len(personas)
    otros = [d for d in detected_unique_es if d != "persona"]
    desc_parts = []
    if n_persons == 1:
        desc_parts.append("Una persona aparece en la imagen.")
    elif n_persons > 1:
        desc_parts.append(f"{n_persons} personas aparecen en la imagen.")
    if otros:
        desc_parts.append(f"Se detectaron: {', '.join(otros[:5])}.")
    desc_parts.append(f"Ambiente {scene['tipo']}, brillo {brightness:.0%}, paleta predominante: {', '.join(color_names[:3])}.")
    descripcion = " ".join(desc_parts) or "Escena analizada localmente por YOLOv8n."

    # Palabras clave
    keywords = list(dict.fromkeys(
        detected_unique_es[:4] + color_names[:2] + [scene["tipo"], img_type]
    ))[:8]

    # Prompt de recreación
    prompt_recreacion = build_prompt_recreacion(detected_unique_es, color_names, scene, brightness)

    # 4. Datos de reconstrucción
    pixel_matrix = build_pixel_matrix(img, sample_step=8)
    thumbnail_b64 = build_thumbnail_b64(img, max_side=128)

    return {
        # ── Descripción semántica ──────────────────────────────────────────
        "sujeto": sujeto_principal,
        "descripcion_semantica": descripcion,
        "tipo_imagen": img_type,
        "personas": personas,
        "objetos_detectados": detected_unique_es,
        "texto_detectado": None,
        "escena": scene,
        "composicion": composition,
        "colores_dominantes": color_names,
        "estado_emocional": "análisis local sin IA generativa",
        "palabras_clave": keywords,
        "prompt_recreacion": prompt_recreacion,

        # ── Detecciones YOLO con coordenadas ──────────────────────────────
        "detecciones_yolo": detections_detail,

        # ── Datos técnicos de imagen ───────────────────────────────────────
        "estructura": {
            "formato": "jpeg",
            "ancho": img.width,
            "alto": img.height,
            "motor": "yolov8n-local"
        },

        # ── Datos para reconstrucción ──────────────────────────────────────
        "reconstruccion": {
            "thumbnail_base64": thumbnail_b64,       # PNG 128px max — cargable en <img>
            "paleta_completa": palette,               # [{hex, rgb}] × 6
            "brillo_normalizado": brightness,
            "saturacion_normalizada": saturation,
            "matriz_pixeles": pixel_matrix            # submuestreada ×8
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "motor": "YOLOv8n local", "endpoint": "POST /extraer-json"}


@app.post("/api/seresinmortales")
def ping():
    return {"message": "Endpoint activo. Motor: YOLOv8n local."}


@app.post("/extraer-json")
async def extraer_json(
    imagen: Optional[UploadFile] = File(None),
    base64: Optional[str] = Form(None),
    mimeType: Optional[str] = Form("image/jpeg")
):
    try:
        # Leer imagen desde multipart o base64
        if imagen is not None:
            raw = await imagen.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif base64 is not None:
            raw_b64 = base64.replace("data:image/jpeg;base64,", "") \
                           .replace("data:image/png;base64,", "") \
                           .replace("data:image/webp;base64,", "")
            raw = __import__("base64").b64decode(raw_b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            raise HTTPException(status_code=400,
                detail='Envía la imagen como multipart (campo "imagen") o base64.')

        result = analyze_image(img)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# ADN Fotográfico — YOLOv8n Local

Sistema completamente local. Sin APIs externas. Sin tokens de pago.

## Arquitectura

```
extraer_json/index.html   (navegador)
        ↓ HTTP POST multipart
Mi_Api_4/index.js         (Node.js, puerto 3000)
        ↓ HTTP POST multipart
yolo_server/yolo_server.py (Python + YOLOv8n, puerto 8000)
```

## Instalación

### 1. Python — servidor YOLOv8n

```bash
pip install fastapi uvicorn ultralytics pillow numpy
```

El modelo `yolov8n.pt` se descarga automáticamente la primera vez que se ejecuta.

### 2. Node.js — Mi_Api_4

```bash
cd Mi_Api_4
npm install
```

Solo usa: `express`, `body-parser`, `cors`, `multer`.  
**Sin** `@anthropic-ai/sdk`, **sin** `@google/generative-ai`.

---

## Arranque (orden importante)

### Terminal 1 — Python YOLOv8n
```bash
cd yolo_server
python yolo_server.py
```
Queda en: `http://localhost:8000`

### Terminal 2 — Node API
```bash
cd Mi_Api_4
node index.js
```
Queda en: `http://localhost:3000`

### Terminal 3 (opcional) — Servidor web para el frontend
```bash
cd extraer_json
# Opción A: Python
python -m http.server 8080

# Opción B: Node
npx serve .
```
Luego abre: `http://localhost:8080`

O simplemente abre `extraer_json/index.html` directamente en el navegador.

---

## Variables de entorno opcionales

| Variable         | Default                    | Descripción                       |
|------------------|----------------------------|-----------------------------------|
| `YOLO_SERVER_URL`| `http://localhost:8000`    | URL del servidor Python YOLOv8n   |
| `PORT`           | `3000`                     | Puerto de Mi_Api_4                |

Ejemplo:
```bash
YOLO_SERVER_URL=http://localhost:8000 PORT=3000 node index.js
```

---

## JSON de salida

El JSON resultante incluye dos bloques:

### Descripción semántica
- `sujeto`: objeto/persona principal detectado
- `descripcion_semantica`: descripción generada automáticamente
- `personas`: lista de personas con bounding box y confianza
- `objetos_detectados`: lista en español de todos los objetos COCO detectados
- `escena`, `composicion`, `colores_dominantes`, `palabras_clave`
- `prompt_recreacion`: prompt en inglés para regenerar con IA generativa
- `detecciones_yolo`: detecciones raw con clase, confianza y bbox exacto

### Reconstrucción
- `reconstruccion.thumbnail_base64`: miniatura PNG 128px en base64 (cargable en `<img>`)
- `reconstruccion.paleta_completa`: 6 colores dominantes con hex y rgb
- `reconstruccion.brillo_normalizado` / `saturacion_normalizada`
- `reconstruccion.matriz_pixeles`: matriz RGB submuestreada ×8 para reconstrucción pixel art

---

## Reconstrucción desde JSON

En la interfaz web, pega el JSON en el panel inferior izquierdo y pulsa **↺ reconstruir**.  
El sistema usa en orden de preferencia:
1. `thumbnail_base64` (imagen fiel, 128px)
2. `matriz_pixeles` (renderizado pixel-art escalable)
3. `paleta_completa` (franjas de color como referencia)

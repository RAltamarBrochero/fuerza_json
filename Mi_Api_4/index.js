/**
 * Mi_Api_4 / index.js
 * ====================
 * API Node.js que actúa como proxy hacia el servidor Python local
 * de YOLOv8n (yolo_server.py corriendo en http://localhost:8000).
 *
 * FLUJO:
 *   extraer_json (web) → Mi_Api_4 (Node, puerto 3000)
 *                      → yolo_server.py (Python, puerto 8000)
 *
 * No usa ninguna API externa ni token de pago.
 *
 * Arranque:
 *   1. python yolo_server.py   (en otra terminal)
 *   2. node index.js            (este archivo)
 */

const express    = require('express');
const bodyParser = require('body-parser');
const multer     = require('multer');
const cors       = require('cors');
const http       = require('http');
const https      = require('https');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '20mb' }));

// URL del servidor Python YOLOv8n local
// Cámbiala si corre en otro puerto
const YOLO_SERVER_URL = process.env.YOLO_SERVER_URL || 'http://localhost:8000';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

// ── Utilidad: reenviar request multipart al servidor Python ──────────────────

function forwardMultipartToPython(imageBuffer, mimeType) {
  return new Promise((resolve, reject) => {
    const boundary = '----FormBoundary' + Math.random().toString(16).slice(2);
    const filename  = 'imagen.jpg';

    // Construir cuerpo multipart manualmente
    const header = Buffer.from(
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="imagen"; filename="${filename}"\r\n` +
      `Content-Type: ${mimeType}\r\n\r\n`
    );
    const footer = Buffer.from(`\r\n--${boundary}--\r\n`);
    const body   = Buffer.concat([header, imageBuffer, footer]);

    const url     = new URL(YOLO_SERVER_URL + '/extraer-json');
    const isHttps = url.protocol === 'https:';
    const lib     = isHttps ? https : http;

    const options = {
      hostname: url.hostname,
      port:     url.port || (isHttps ? 443 : 80),
      path:     url.pathname,
      method:   'POST',
      headers: {
        'Content-Type':   `multipart/form-data; boundary=${boundary}`,
        'Content-Length': body.length
      }
    };

    const req = lib.request(options, (res) => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, body: { error: 'Respuesta no JSON del servidor Python', raw: data.slice(0, 300) } });
        }
      });
    });

    req.on('error', (err) => {
      reject(new Error(`No se pudo conectar al servidor YOLOv8n en ${YOLO_SERVER_URL}. ` +
        '¿Está corriendo yolo_server.py? Error: ' + err.message));
    });

    req.write(body);
    req.end();
  });
}

// ── Health check del servidor Python ─────────────────────────────────────────

function checkPythonServer() {
  return new Promise((resolve) => {
    const url = new URL(YOLO_SERVER_URL + '/');
    const lib = url.protocol === 'https:' ? https : http;
    const req = lib.get({ hostname: url.hostname, port: url.port || 80, path: '/' }, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(3000, () => { req.destroy(); resolve(false); });
  });
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

// Ping / health check (para que el frontend detecte si hay key configurada)
app.get('/api/check-key', async (req, res) => {
  const pythonOk = await checkPythonServer();
  res.json({
    configurada: pythonOk,
    motor: 'YOLOv8n local',
    yolo_server: YOLO_SERVER_URL,
    yolo_server_activo: pythonOk
  });
});

// Compatibilidad con endpoint antiguo
app.post('/api/seresinmortales', (req, res) => {
  res.status(200).json({ message: 'Endpoint activo. Motor: YOLOv8n local.' });
});

// Endpoint principal
app.post('/api/extraer-json', upload.single('imagen'), async (req, res) => {
  try {
    let imageBuffer, mediaType;

    if (req.file) {
      // Imagen como multipart
      imageBuffer = req.file.buffer;
      mediaType   = req.file.mimetype;
    } else if (req.body && req.body.base64) {
      // Imagen como base64
      const raw   = req.body.base64.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(raw, 'base64');
      mediaType   = req.body.mimeType || 'image/jpeg';
    } else {
      return res.status(400).json({
        error: 'Envía la imagen como multipart (campo "imagen") o como base64.'
      });
    }

    // Reenviar al servidor Python YOLOv8n
    const { status, body } = await forwardMultipartToPython(imageBuffer, mediaType);
    res.status(status).json(body);

  } catch (err) {
    console.error('[Mi_Api_4]', err.message);
    res.status(503).json({
      error: err.message,
      hint: `Asegúrate de que yolo_server.py esté corriendo en ${YOLO_SERVER_URL}`
    });
  }
});

// Iniciar servidor
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n✅ Mi_Api_4 corriendo en http://localhost:${PORT}`);
  console.log(`   Motor: YOLOv8n local → ${YOLO_SERVER_URL}`);
  console.log(`   Sin dependencias externas ni tokens de pago.\n`);
});

/**
 * router.js — Exporta las rutas de Mi_Api_4 como un Express Router.
 * Usado por unir.js para montar la API bajo /api sin levantar un servidor separado.
 * El index.js original sigue funcionando igual (node index.js en puerto 5000).
 */

const express    = require('express');
const bodyParser = require('body-parser');
const multer     = require('multer');
const Anthropic  = require('@anthropic-ai/sdk');
const cors       = require('cors');

const router = express.Router();

router.use(cors());
router.use(bodyParser.json({ limit: '20mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

// ── Prompt semántico (idéntico al de index.js) ─────────────────────────────
const PROMPT_SEMANTICA = `Analiza esta imagen y devuelve ÚNICAMENTE un JSON válido (sin texto adicional, sin backticks, sin explicaciones) con exactamente esta estructura:

{
  "sujeto": "qué o quién es el elemento principal de la imagen",
  "descripcion_semantica": "descripción visual completa y detallada en 2-3 oraciones",
  "tipo_imagen": "foto|arte_digital|ilustracion|captura_pantalla|documento",
  "personas": [
    {
      "descripcion": "descripción física",
      "genero_aparente": "masculino|femenino|no_determinado",
      "edad_aproximada": "rango en años",
      "expresion": "descripción de expresión facial",
      "ropa": "descripción de la ropa",
      "accion": "qué está haciendo"
    }
  ],
  "objetos_detectados": ["lista de todos los objetos visibles"],
  "texto_detectado": "cualquier texto visible en la imagen, o null si no hay",
  "escena": {
    "tipo": "interior|exterior|natural|urbano|estudio",
    "lugar_probable": "descripción del lugar",
    "momento_dia": "dia|noche|atardecer|amanecer|no_determinado",
    "clima": "soleado|nublado|lluvioso|no_aplica",
    "ambiente": "descripción del ambiente general"
  },
  "composicion": {
    "plano": "primer_plano|plano_medio|plano_general|plano_detalle|panoramico",
    "angulo": "frontal|lateral|cenital|contrapicado|perspectiva"
  },
  "colores_dominantes": ["color1", "color2", "color3"],
  "estado_emocional": "descripción del tono emocional o atmósfera de la imagen",
  "palabras_clave": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8"],
  "prompt_recreacion": "prompt detallado en inglés para recrear esta imagen con IA generativa"
}
Si no hay personas deja personas:[]. Responde SOLO con el JSON, nada más.`;

// ── Rutas ──────────────────────────────────────────────────────────────────

// POST /api/seresinmortales
router.post('/api/seresinmortales', (req, res) => {
  res.status(200).json({ message: 'Endpoint activo.' });
});

// POST /api/extraer-json
router.post('/extraer-json', upload.single('imagen'), async (req, res) => {
  try {
    const apiKey = req.headers['x-api-key'] || process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return res.status(401).json({ error: 'Falta la Anthropic API Key. Envíala en el header X-Api-Key.' });
    }

    const anthropic = new Anthropic({ apiKey });

    let imageBuffer, mediaType;
    if (req.file) {
      imageBuffer = req.file.buffer;
      mediaType   = req.file.mimetype;
    } else if (req.body.base64) {
      const raw   = req.body.base64.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(raw, 'base64');
      mediaType   = req.body.mimeType || 'image/jpeg';
    } else {
      return res.status(400).json({ error: 'Envía una imagen como multipart (campo "imagen") o como base64.' });
    }

    const base64  = imageBuffer.toString('base64');
    const formato = mediaType.split('/')[1] || 'jpeg';
    const { ancho, alto } = getImageDimensions(imageBuffer, formato);

    const response = await anthropic.messages.create({
      model: 'claude-opus-4-6',
      max_tokens: 1500,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64 } },
          { type: 'text', text: PROMPT_SEMANTICA }
        ]
      }]
    });

    let semantica;
    try {
      const rawText = response.content[0].text.trim();
      const cleaned = rawText.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();
      semantica = JSON.parse(cleaned);
    } catch {
      semantica = { error: 'No se pudo parsear la respuesta de Claude' };
    }

    semantica.estructura = { formato, ancho, alto };
    res.json(semantica);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Error interno' });
  }
});

// ── Helpers ────────────────────────────────────────────────────────────────
function getImageDimensions(buffer, formato) {
  try {
    if (formato === 'png') {
      return { ancho: buffer.readUInt32BE(16), alto: buffer.readUInt32BE(20) };
    }
    if (formato === 'jpeg' || formato === 'jpg') {
      let i = 2;
      while (i < buffer.length) {
        const marker = buffer.readUInt16BE(i); i += 2;
        if (marker >= 0xFFC0 && marker <= 0xFFC3) {
          return { alto: buffer.readUInt16BE(i + 3), ancho: buffer.readUInt16BE(i + 5) };
        }
        i += buffer.readUInt16BE(i);
      }
    }
  } catch {}
  return { ancho: null, alto: null };
}

module.exports = router;

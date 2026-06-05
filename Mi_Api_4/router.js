/**
 * router.js — Rutas de Mi_Api_4 usando Google Gemini (gratuito)
 * Montado por unir.js bajo /api
 * La API Key se lee desde: variable de entorno GEMINI_API_KEY o header X-Api-Key
 */

const express    = require('express');
const bodyParser = require('body-parser');
const multer     = require('multer');
const cors       = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const router = express.Router();
router.use(cors());
router.use(bodyParser.json({ limit: '20mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

const PROMPT_SEMANTICA = `Analiza esta imagen y devuelve UNICAMENTE un JSON valido (sin texto adicional, sin backticks, sin explicaciones) con exactamente esta estructura:
{"sujeto":"","descripcion_semantica":"","tipo_imagen":"foto|arte_digital|ilustracion|captura_pantalla|documento","personas":[{"descripcion":"","genero_aparente":"masculino|femenino|no_determinado","edad_aproximada":"","expresion":"","ropa":"","accion":""}],"objetos_detectados":[],"texto_detectado":null,"escena":{"tipo":"interior|exterior|natural|urbano|estudio","lugar_probable":"","momento_dia":"dia|noche|atardecer|amanecer|no_determinado","clima":"soleado|nublado|lluvioso|no_aplica","ambiente":""},"composicion":{"plano":"primer_plano|plano_medio|plano_general|plano_detalle|panoramico","angulo":"frontal|lateral|cenital|contrapicado|perspectiva"},"colores_dominantes":[],"estado_emocional":"","palabras_clave":[],"prompt_recreacion":""}
Si no hay personas deja personas:[]. Responde SOLO con el JSON.`;

router.post('/api/seresinmortales', (req, res) => {
  res.status(200).json({ message: 'Endpoint activo.' });
});

router.post('/extraer-json', upload.single('imagen'), async (req, res) => {
  try {
    const apiKey = req.headers['x-api-key']
                || process.env.GEMINI_API_KEY
                || process.env.GOOGLE_API_KEY;

    if (!apiKey) {
      return res.status(401).json({ error: 'Falta la Gemini API Key. Configura GEMINI_API_KEY en el sistema o ingrésala en la web.' });
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

    let imageBuffer, mediaType;
    if (req.file) {
      imageBuffer = req.file.buffer;
      mediaType   = req.file.mimetype;
    } else if (req.body.base64) {
      const raw   = req.body.base64.replace(/^data:image\/\w+;base64,/, '');
      imageBuffer = Buffer.from(raw, 'base64');
      mediaType   = req.body.mimeType || 'image/jpeg';
    } else {
      return res.status(400).json({ error: 'Envia una imagen como multipart o base64.' });
    }

    const base64  = imageBuffer.toString('base64');
    const formato = mediaType.split('/')[1] || 'jpeg';
    const { ancho, alto } = getImageDimensions(imageBuffer, formato);

    const result = await model.generateContent([
      PROMPT_SEMANTICA,
      { inlineData: { mimeType: mediaType, data: base64 } }
    ]);

    const rawText = result.response.text().trim();
    const cleaned = rawText.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();

    let semantica;
    try {
      semantica = JSON.parse(cleaned);
    } catch {
      semantica = { error: 'No se pudo parsear la respuesta de Gemini', raw: cleaned.slice(0, 300) };
    }

    semantica.estructura = { formato, ancho, alto, motor: 'gemini-2.0-flash' };
    res.json(semantica);

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Error interno' });
  }
});

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

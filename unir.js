/**
 * unir.js — Servidor principal
 * Une la API (Mi_Api_4) y la web (extraer_json) en un solo proceso.
 * Ejecutar desde la carpeta fuerza_json:  node unir.js
 */

const express = require('express');
const path    = require('path');

// ── Cambia el cwd a Mi_Api_4 ANTES de requerir su módulo,
//    para que database.js resuelva ./seresInmortales.db correctamente.
const API_DIR = path.join(__dirname, 'Mi_Api_4');
process.chdir(API_DIR);

// ── Cargar la lógica de la API como un router reutilizable
const apiRouter = require('./Mi_Api_4/router');

// ── Volver al directorio raíz del proyecto
process.chdir(__dirname);

// ── App principal
const app = express();

// Archivos estáticos de la web (extraer_json/index.html)
app.use(express.static(path.join(__dirname, 'extraer_json')));

// Endpoint para que la web sepa si el servidor ya tiene la API key configurada
app.get('/api/check-key', (req, res) => {
  const tieneKey = !!process.env.ANTHROPIC_API_KEY;
  res.json({ configurada: tieneKey });
});

// Montar todas las rutas de la API bajo /api
//   /extraer-json      → /api/extraer-json
//   /api/seresinmortales → /api/api/seresinmortales  ← evitar doble prefijo
app.use('/api', apiRouter);

// Ruta raíz → sirve la web
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'extraer_json', 'index.html'));
});

// Iniciar servidor
const PORT = 3000;
app.listen(PORT, () => {
  console.log('');
  console.log('  ✓ Servidor unificado corriendo en http://localhost:' + PORT);
  console.log('');
  console.log('  → Web:   http://localhost:' + PORT + '/index.html');
  console.log('  → API:   http://localhost:' + PORT + '/api/extraer-json');
  console.log('           http://localhost:' + PORT + '/api/api/seresinmortales');
  console.log('');
  console.log('  En la web, cambia "URL DE MI API" a:');
  console.log('  http://localhost:' + PORT + '/api/extraer-json');
  console.log('');
});

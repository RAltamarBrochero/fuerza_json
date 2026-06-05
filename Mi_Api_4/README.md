# Mi_Api_4 — API de Visión con Claude

API REST en Node.js + Express que procesa imágenes con **Claude Vision** y devuelve un JSON semántico estructurado.

## Estructura del proyecto

```
Mi_Api_4/
├── index.js          # API principal (Express)
├── database.js       # Conexión SQLite (seres inmortales)
├── index.html        # Frontend — extractor JSON visual
├── package.json      # Dependencias
└── seresInmortales.db
```

## Instalación

```bash
npm install
```

## Uso

```bash
export ANTHROPIC_API_KEY=sk-ant-...
node index.js
```

El servidor arranca en `http://localhost:5000`.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/extraer-json` | Recibe imagen → devuelve JSON semántico + estructural |
| `GET`  | `/api/seresinmortales` | Lista todos los seres |
| `GET`  | `/api/seresinmortales/:id` | Obtiene un ser por ID |
| `POST` | `/api/seresinmortales` | Crea un nuevo ser |
| `DELETE` | `/api/seresinmortales/:id` | Elimina un ser |

## Ejemplo de respuesta `/extraer-json`

```json
{
  "estructura": {
    "formato": "jpeg",
    "ancho": 1280,
    "alto": 720,
    "base64": "..."
  },
  "semantica": {
    "personas": [
      { "id": 1, "accion": "camina", "emocion": "neutro" }
    ],
    "objetos": ["mesa", "silla"],
    "escena": { "tipo": "interior", "momento": "día" }
  }
}
```

## Frontend

Abre `index.html` en el navegador. Pon `http://localhost:5000/extraer-json` en el campo URL, sube una imagen y selecciona una región para analizarla.

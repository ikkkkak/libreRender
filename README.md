# MarianMT Translation Service

Self-hosted translation API using Hugging Face MarianMT models.

## Deployment on Render

### Environment Variables
- `PORT` - Automatically set by Render (defaults to 10000 for local dev)

### Health Check
- `GET /` or `GET /health` - Returns service status

### API Endpoint
- `POST /translate` - Translate text

### Request Format
```json
{
  "text": "Hello world",
  "source_lang": "en",
  "target_lang": "ar"
}
```

### Response Format
```json
{
  "translated_text": "مرحبا بالعالم"
}
```

### Supported Language Pairs
- en ↔ fr
- en ↔ ar
- fr ↔ ar

### Notes
- Models are loaded on-demand and cached in memory
- First request for each language pair may be slow (model download)
- Render cold starts may take up to 60 seconds


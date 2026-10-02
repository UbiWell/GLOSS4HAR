# GLOSS4HAR

Sensemaking over passive-sensing data with LLM agents (GLOSS) applied to human activity recognition.

## Configuration

Credentials and endpoints are read from environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI models (gpt-4o / gpt-5) | — |
| `LLM_API_URL` | gpt-oss endpoint (Ollama) | `http://localhost:11434/api/generate` |
| `LLM_MODEL` | gpt-oss model name | `gpt-oss:20b` |
| `MONGO_HOST`, `MONGO_PORT`, `MONGO_DB` | MongoDB holding the sensor data | `localhost`, `27017`, `gloss` |
| `MONGO_USER`, `MONGO_PASSWORD` | MongoDB credentials | empty |
| `GOOGLE_API_KEY` | Geocoding for location tools | empty |

Model backend and ablation flags are set in `agents/constants.py`.

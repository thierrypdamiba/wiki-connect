# Wiki Connect

Connect any two topics using Qdrant vector search across 35M Wikipedia articles, with Linkup for real-time web grounding.

## Stack

- **Qdrant** - Vector search (35M Wikipedia articles, user memory)
- **Linkup** - Real-time web search fallback
- **Google ADK** - Agent framework with Gemini 2.0 Flash
- **Next.js** - Frontend

## Setup

```bash
# Backend
uv sync

# Frontend
cd frontend && pnpm install
```

Create `.env`:
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key
LINKUP_API_KEY=your_linkup_key
```

## Run

```bash
# Backend
uv run uvicorn api_server:app --port 8000

# Frontend
cd frontend && pnpm dev
```

Open http://localhost:3000

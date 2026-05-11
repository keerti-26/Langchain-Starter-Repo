# LangChain Starter Agent

A minimal FastAPI app with LangChain agent endpoints powered by [LangGraph](https://langchain-ai.github.io/langgraph/), now with a lightweight browser frontend.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat/tools` | **Tool Agent** — stateless, has calculator, current time, and knowledge base tools |
| `POST` | `/chat/memory` | **Memory Agent** — stateful conversational agent with per-session memory |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Run the backend server
uvicorn app.main:app --reload
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Frontend (New)

A simple single-page chat UI is available at `frontend/index.html`.

### Run Frontend Locally

```bash
# from project root
python -m http.server 5173 --directory frontend
```

Then open [http://localhost:5173](http://localhost:5173).

Notes:
- Backend must still be running on `http://localhost:8000`.
- CORS is preconfigured in FastAPI for `localhost:5173` and `127.0.0.1:5173`.
- In the UI, choose:
  - **Tool Agent** for stateless calls to `/chat/tools`
  - **Memory Agent** for stateful calls to `/chat/memory` (session_id is auto-filled after first message)

## Usage Examples

### Tool Agent

```bash
# Calculator
curl -X POST http://localhost:8000/chat/tools \
  -H "Content-Type: application/json" \
  -d '{"message": "What is sqrt(144) + 10?"}'

# Current time
curl -X POST http://localhost:8000/chat/tools \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it right now?"}'

# Knowledge base search
curl -X POST http://localhost:8000/chat/tools \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the refund policy?"}'
```

### Memory Agent

```bash
# First message — returns a session_id
curl -X POST http://localhost:8000/chat/memory \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Zach and I like Python."}'

# Follow-up — pass the session_id to continue the conversation
curl -X POST http://localhost:8000/chat/memory \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my name?", "session_id": "<session_id_from_above>"}'
```

## Project Structure

```text
langchain-start-repo/
├── app/
│   ├── __init__.py
│   ├── config.py       # LLM configuration
│   ├── tools.py        # Custom tool definitions
│   ├── agents.py       # Agent builders (tool + memory)
│   ├── schemas.py      # Pydantic request/response models
│   └── main.py         # FastAPI app and endpoints (+ CORS for frontend)
├── frontend/
│   └── index.html      # Lightweight chat UI
├── .env.example
├── requirements.txt
└── README.md
```

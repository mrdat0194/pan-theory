# Part B — EDB Policy Q&A Endpoint

## Setup

```bash
cd partB
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
uvicorn app:app --reload
```

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `TOP_K` | Number of chunks to retrieve per query (default: 4) |

## API

### POST /ask

Request:
```json
{"question": "What is the Enterprise Development Grant?"}
```

Response:
```json
{
  "answer": "...",
  "sources": ["enterprise_development_grant.txt — Overview section..."]
}
```

### GET /health

Returns `{"status": "ok", "vectorstore_ready": true}` if the vector store initialised successfully.

---

## Design decisions

<!-- Complete this section — explain at least ONE deliberate design choice -->

### Chunk size and overlap
*Your explanation here.*
e.g.: "I chose chunk_size=500 with overlap=50 because..."

### Prompt design
*Your explanation here.*

---

## Test query outputs

Paste the raw JSON responses for both test queries below.

### Answerable query
```
Question: "What support does EDB offer for workforce development?"
Response:
```

### Out-of-scope query
```
Question: "What is the current Singapore prime minister's salary?"
Response:
```

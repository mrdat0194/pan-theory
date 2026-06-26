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

### Chunk size and overlap
I chose `chunk_size=500` with `overlap=50`. This size is large enough to capture complete sentences and meaningful context from the policy documents, which are relatively short. The overlap ensures that if a key piece of information is split between two chunks, it can still be retrieved and understood in context.

### Prompt design
The system prompt explicitly instructs the model to:
1. Act as an expert assistant for EDB policy questions.
2. Use only the provided context to answer questions.
3. Cite the source document filename for every claim made.
4. Say "I don't know" if the answer is not in the context.
This ensures grounding and prevents hallucinations.

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

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
I chose `chunk_size=500` with `chunk_overlap=50`. This size is large enough to capture meaningful context (around 100-150 words) while remaining small enough to fit multiple chunks into the LLM prompt without exceeding token limits. The 50-character overlap helps ensure that sentences or concepts aren't abruptly cut off between chunks, maintaining semantic continuity.

### Prompt design
The prompt is designed as a `ChatPromptTemplate` with a clear system instruction. It enforces the "answer using ONLY the provided context" constraint and specifically instructs the model to say "I don't know" if the information is missing. The context is presented with source headers to help the model distinguish between different documents, although the final citation formatting is handled programmatically in the endpoint logic to ensure consistent "filename — preview" output as requested.

---

## Test query outputs

### Answerable query
```
Question: "What support does EDB offer for workforce development?"
Response:
{
  "answer": "EDB offers several support measures for workforce development:\n1. **Talent Identification and Upskilling**: EDB works with companies to identify talent needs and develop programmes to upskill employees.\n2. **Leadership Development Programme (LDP)**: This programme aims to help companies develop their next generation of leaders.\n3. **SkillsFuture Enterprise Credit (SFEC)**: This credit encourages employers to invest in enterprise transformation and the capabilities of their employees.",
  "sources": [
    "workforce_development.txt — EDB offers various support for workforce development. We work with companies to identify their talen...",
    "enterprise_development_grant.txt — The Enterprise Development Grant (EDG) helps Singapore companies grow and transform. This grant suppor..."
  ]
}
```

### Out-of-scope query
```
Question: "What is the current Singapore prime minister's salary?"
Response:
{
  "answer": "I don't know.",
  "sources": [
    "workforce_development.txt — EDB offers various support for workforce development. We work with companies to identify their talen...",
    "innovation_and_ip.txt — Innovation and Intellectual Property (IP) are critical for business competitiveness. EDB provides sup...",
    "enterprise_development_grant.txt — The Enterprise Development Grant (EDG) helps Singapore companies grow and transform. This grant suppor...",
    "innovation_and_ip.txt — The Intellectual Property Development Programme helps companies protect and leverage their IP assets f..."
  ]
}
```

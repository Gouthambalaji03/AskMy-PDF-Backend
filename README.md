# AskMyPDF Backend

REST + SSE API server for AskMyPDF. Handles PDF processing, hybrid search (vector + keyword), and streams AI-generated answers using Google Gemini.

## Tech Stack

- **Runtime**: Node.js + TypeScript
- **Framework**: Express.js
- **Database**: MongoDB Atlas (documents, chunks, chat history, query cache)
- **Search**: Atlas Vector Search + Atlas Text Search with brute-force fallbacks
- **AI**: Google Gemini (`gemini-embedding-001` for embeddings, `gemini-2.5-flash` / `gemini-2.0-flash` for chat)
- **PDF Parsing**: pdf-parse

## Architecture

```
Request
  |
  +--> pdfRoutes.ts (API layer)
  |       |
  |       +--> pdfService.ts      -- PDF extraction + recursive chunking
  |       +--> embeddingService.ts -- Gemini embeddings + re-ranking
  |       +--> chatService.ts     -- LLM chat with streaming + fallback
  |       +--> vectorStore.ts     -- MongoDB hybrid search (vector + keyword)
  |
  +--> models/
          +--> Document.ts  -- PDF metadata
          +--> Chunk.ts     -- Text + embedding per chunk
          +--> Chat.ts      -- Q&A history per document
          +--> QueryCache.ts -- Cached answers (24h TTL)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload and process a PDF |
| `POST` | `/api/ask` | Ask a question (non-streaming) |
| `POST` | `/api/ask/stream` | Ask a question (SSE streaming) |
| `GET` | `/api/documents` | List all uploaded documents |
| `GET` | `/api/documents/:id/chat` | Get chat history for a document |
| `DELETE` | `/api/documents/:id` | Delete a document and all related data |
| `GET` | `/health` | Health check + database status |

## How the RAG Pipeline Works

```
1. Upload: PDF --> extract text --> recursive chunk (paragraphs > sentences > fixed-size)
                                --> generate 3072-dim embeddings via Gemini
                                --> store chunks + embeddings in MongoDB

2. Ask:    Question --> embed question
                   --> hybrid search (vector + keyword via MongoDB Atlas)
                   --> merge results with Reciprocal Rank Fusion (RRF)
                   --> re-rank with 4 signals (hybrid score, coverage, proximity, exact match)
                   --> build prompt with top 3 chunks + last 5 Q&A turns
                   --> stream answer from Gemini via SSE
                   --> cache result for 24 hours
```

## Features

- **Hybrid Search** -- Vector (semantic) + keyword (BM25) merged via RRF (k=60)
- **Re-ranking** -- 4-signal scoring: hybrid score (35%), term coverage (25%), proximity (20%), exact match (20%)
- **Streaming (SSE)** -- Real-time token streaming with client disconnect detection
- **Conversational Memory** -- Last 5 Q&A pairs sent as context for follow-up questions
- **Query Caching** -- SHA-256 hashed cache key, 24h TTL auto-expiry
- **Model Fallback** -- gemini-2.5-flash --> gemini-2.0-flash with exponential backoff on 429/503
- **Rate Limiting** -- 15 asks/min, 5 uploads/min per IP
- **Input Validation** -- Question length (1000 chars), ObjectId format, file type/size

## Setup

### Prerequisites

- Node.js 18+
- MongoDB Atlas cluster with:
  - Vector index `chunk_vector_index` on `chunks` collection
  - Text index `chunk_text_index` on `chunks` collection
- Google Gemini API key

### Install

```bash
git clone <repo-url>
cd AskMy-PDF-Backend
npm install
```

### Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/askmy-PDF
FRONTEND_URL=http://localhost:3000
PORT=5000
MAX_FILE_SIZE=10485760
```

### Atlas Search Indexes

Create these indexes in MongoDB Atlas on the `chunks` collection:

**Vector index** (`chunk_vector_index`):
```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 3072, "similarity": "cosine" },
    { "type": "filter", "path": "documentId" }
  ]
}
```

**Text index** (`chunk_text_index`):
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": { "type": "string", "analyzer": "lucene.standard" },
      "documentId": { "type": "objectId" }
    }
  }
}
```

> Without these indexes, the app falls back to brute-force search (slower but functional).

### Run

```bash
# Development (hot-reload)
npm run dev

# Production
npm run build
npm start
```

### Deploy (Render)

1. Push to GitHub
2. Create a new Web Service on Render
3. Set build command: `npm install && npm run build`
4. Set start command: `node dist/index.js`
5. Add environment variables in Render dashboard

## Project Structure

```
src/
  config/
    database.ts         -- MongoDB connection
  models/
    Document.ts         -- PDF metadata schema
    Chunk.ts            -- Text + embedding schema
    Chat.ts             -- Q&A history schema
    QueryCache.ts       -- Cache with 24h TTL
  routes/
    pdfRoutes.ts        -- All API endpoints
  services/
    pdfService.ts       -- PDF extraction + chunking
    embeddingService.ts -- Gemini embeddings + re-ranking
    chatService.ts      -- LLM chat + streaming + fallback
  store/
    vectorStore.ts      -- MongoDB hybrid search
  index.ts              -- Server entry point
```

## License

MIT

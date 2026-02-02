# System Overview & Architecture

A comprehensive technical document covering the complete Personal RAG (Retrieval-Augmented Generation) system architecture, technologies, data flows, API specifications, and implementation details.

---

## 1. High-Level Architecture

The Personal RAG system is a document ingestion and semantic search platform that transforms unstructured documents into queryable vector embeddings. The architecture follows a modular ETL (Extract-Transform-Load) pattern with asynchronous job processing.

### Architecture Diagram (Conceptual)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌──────────────┐                                                           │
│  │   Frontend   │  Hosted on Vercel                                         │
│  │   (Future)   │  Makes API calls to backend                               │
│  └──────┬───────┘                                                           │
└─────────┼───────────────────────────────────────────────────────────────────┘
          │ HTTP/REST
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Application                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ POST /ingest│  │GET /ingest/ │  │  /health    │  │  /debug/*   │  │   │
│  │  │             │  │  {job_id}   │  │             │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Background Task Queue                             │   │
│  │           Semaphore-limited (1 concurrent job)                        │   │
│  │                  In-memory job store                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                   │
│                                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │  CHUNKING  │───▶│  ROUTING   │───▶│ EMBEDDING  │───▶│ UPSERTING  │       │
│  │  (Docling) │    │(OpenRouter)│    │  (OpenAI)  │    │ (Pinecone) │       │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Pinecone Vector Database                         │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │  personal_life  │  │professional_life│  │   about_rag     │       │   │
│  │  │   (namespace)   │  │   (namespace)   │  │   (namespace)   │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow Summary

1. **Ingress**: Client uploads document(s) via multipart form POST
2. **Validation**: File type, namespace, index, and routing mode validation
3. **Queuing**: Job created and added to background task queue
4. **Response**: Immediate 202 Accepted with job ID(s)
5. **Processing**: Async ETL pipeline (chunk → route → embed → upsert)
6. **Storage**: Vectors stored in Pinecone with rich metadata
7. **Cleanup**: Temporary files removed on success

---

## 2. Technology Stack

### Complete Technology Matrix

| Layer | Component | Technology | Version/Details |
|-------|-----------|------------|-----------------|
| **Runtime** | Language | Python | 3.11 |
| **Runtime** | Base Image | python:3.11-slim | Debian-based |
| **Framework** | Web Framework | FastAPI | Latest |
| **Framework** | ASGI Server | Uvicorn | With standard extras |
| **Parsing** | Document Parser | Docling (IBM) | HybridChunker |
| **Parsing** | OCR Engine | Docling built-in | PDF text extraction |
| **Parsing** | Image Processing | opencv-python-headless | Headless for containers |
| **Parsing** | Tokenizer | tiktoken / gpt2 | Configurable |
| **Embedding** | Embedding API | OpenAI | text-embedding-3-small |
| **Embedding** | Vector Dimensions | 1536 | Fixed |
| **Routing** | LLM Provider | OpenRouter | Free tier models |
| **Routing** | Default Model | meta-llama/llama-3.2-3b-instruct | :free suffix |
| **Storage** | Vector Database | Pinecone | gRPC client |
| **Storage** | Temporary Files | Local filesystem | /tmp/rag-uploads/ |
| **Validation** | Data Validation | Pydantic | v2 |
| **Validation** | Settings | pydantic-settings | Environment loading |
| **HTTP** | HTTP Client | httpx | Async capable |
| **Resilience** | Retry Library | tenacity | Exponential backoff |
| **Data** | Data Processing | pandas | CSV/JSON handling |
| **Data** | Numerical | numpy | Array operations |
| **Container** | Containerization | Docker | Multi-stage optional |
| **Container** | Orchestration | Docker Compose | Dev environment |
| **Hosting** | Backend | Railway | Production |
| **Hosting** | Frontend | Vercel | Future |

### System Dependencies

| Package | Purpose |
|---------|---------|
| `libglib2.0-0` | GLib library for Docling's layout engine |

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework with automatic OpenAPI docs |
| `uvicorn[standard]` | ASGI server with uvloop and httptools |
| `python-multipart` | Multipart form data parsing for file uploads |
| `pydantic` | Data validation and serialization |
| `pydantic-settings` | Configuration management from environment |
| `python-dotenv` | .env file loading |
| `openai` | OpenAI API client for embeddings |
| `pinecone` | Pinecone vector database client |
| `protobuf` | Protocol buffers for Pinecone gRPC |
| `grpcio` | gRPC support for Pinecone |
| `googleapis-common-protos` | Google proto definitions |
| `httpx` | Modern async HTTP client for OpenRouter |
| `tenacity` | Retry library with decorators |
| `tiktoken` | Token counting for OpenAI models |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation for CSV/JSON parsing |
| `opencv-python-headless` | Image processing without GUI dependencies |
| `docling` | IBM's document parsing and OCR library |

---

## 3. Core Modules & Directory Structure

### Project Structure

```
personal-RAG/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── ingest.py              # Ingestion API router
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Settings and configuration
│   │   ├── logging.py             # Logging infrastructure
│   │   └── namespaces.py          # Namespace definitions
│   ├── models/
│   │   ├── __init__.py
│   │   └── ingest.py              # Pydantic data models
│   └── services/
│       ├── __init__.py
│       ├── embedder.py            # OpenAI embedding service
│       ├── file_storage.py        # File upload/cleanup
│       ├── ingest_queue.py        # Job queue and ETL orchestration
│       ├── namespace_router.py    # LLM-based classification
│       ├── parser.py              # Document parsing with Docling
│       └── vectordb.py            # Pinecone operations
├── tests/
│   └── test_ingest_queue.py       # Unit tests
├── docs/
│   ├── api_specs/
│   │   └── ingestion.openapi.yaml # OpenAPI specification
│   ├── architecture/
│   │   └── system_overview.md     # This document
│   └── guides/
│       └── ingestion_setup.md     # Setup guide
├── Dockerfile                      # Production container
├── Dockerfile.dev                  # Development container
├── docker-compose.dev.yml          # Dev orchestration
├── requirements.txt                # Python dependencies
├── my.env                          # Environment variables (gitignored)
└── README.md
```

### Module Responsibilities

#### `app/main.py` - Application Entry Point
- FastAPI application instantiation
- Router registration
- Startup event handlers
- Health check endpoint
- Logging initialization

#### `app/api/ingest.py` - Ingestion API Router
- POST `/v1/ingest` - Document upload endpoint
- GET `/v1/ingest/{job_id}` - Job status endpoint
- POST `/v1/debug/reset` - Clear cached clients
- POST `/v1/debug/test-openai` - Test OpenAI connection
- Request validation and error handling
- Background task scheduling

#### `app/core/config.py` - Configuration Management
- Environment variable loading from `my.env`
- Settings dataclass with validation
- Pinecone host resolution (global and per-index)
- Cached settings with reset capability

#### `app/core/logging.py` - Logging Infrastructure
- Centralized logging configuration
- Structured log format with timestamps
- Per-module logger factory

#### `app/core/namespaces.py` - Namespace Definitions
- Namespace enum (PERSONAL_LIFE, PROFESSIONAL_LIFE, ABOUT_RAG)
- Namespace validation utilities
- LLM classification prompt generation

#### `app/models/ingest.py` - Data Models
- RoutingMode enum
- IngestRequest model
- IngestJobSummary model
- IngestAccepted response model
- IngestJobRecord model with status tracking

#### `app/services/ingest_queue.py` - Job Queue & ETL
- In-memory job store (JOB_STORE dict)
- Job queue (JOB_QUEUE list)
- Thread-safe operations (JOB_STORE_LOCK)
- Concurrency control (asyncio Semaphore)
- Full ETL pipeline orchestration
- Status updates at each stage

#### `app/services/parser.py` - Document Parsing
- Multi-format support (PDF, DOCX, MD, TXT, JSON, CSV)
- Docling integration with HybridChunker
- Metadata enrichment (page numbers, headings)
- Contextualization for embeddings
- Fallback handling for simple documents

#### `app/services/embedder.py` - Embedding Service
- OpenAI client management
- Batch embedding with rate limiting
- Retry logic for rate limit errors
- Inter-batch delays for API compliance

#### `app/services/namespace_router.py` - LLM Classification
- OpenRouter HTTP client
- Classification prompt building
- Document-level and per-chunk routing
- Fallback handling with tracking

#### `app/services/vectordb.py` - Vector Database
- Pinecone gRPC client
- Batch upsert operations
- Retry logic for connection errors
- Namespace-based organization

#### `app/services/file_storage.py` - File Management
- Temporary file storage
- Job-isolated directories
- Cleanup on successful processing

---

## 4. Configuration & Environment Variables

### Required Environment Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `OPENAI_API_KEY` | string | OpenAI API key for embedding generation | `sk-proj-...` |
| `OPENROUTER_API_KEY` | string | OpenRouter API key for LLM classification | `sk-or-...` |
| `PINECONE_API_KEY` | string | Pinecone API key for vector operations | `pcsk_...` |
| `PINECONE_INDEX` | string | Default Pinecone index name | `personal-rag-index` |

### Optional Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENROUTER_BASE_URL` | string | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `OPENROUTER_MODEL` | string | `meta-llama/llama-3.2-3b-instruct:free` | LLM model for classification |
| `PINECONE_HOST` | string | None | Global Pinecone host override |
| `PINECONE_HOST_{INDEX}` | string | None | Per-index Pinecone host (e.g., `PINECONE_HOST_MY_INDEX`) |
| `DOCLING_TOKENIZER` | string | `gpt2` | Tokenizer for HybridChunker |

### Configuration Loading Order

1. Load from `my.env` file (if exists)
2. Environment variables override file values
3. Settings cached globally after first load
4. Call `reset_settings()` to reload

### Settings Dataclass

```
Settings:
  openai_api_key: str          # Required
  openrouter_api_key: str      # Required
  pinecone_api_key: str        # Required
  pinecone_index: str          # Required
  openrouter_base_url: str     # Default: https://openrouter.ai/api/v1
  openrouter_model: str        # Default: meta-llama/llama-3.2-3b-instruct:free
  pinecone_host: str | None    # Optional global host
  docling_tokenizer: str       # Default: gpt2
```

---

## 5. API Endpoints

### Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://<railway-app>.railway.app`

### Endpoint Summary

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | `/v1/ingest` | Upload and process documents | None |
| GET | `/v1/ingest/{job_id}` | Get job status | None |
| POST | `/v1/debug/reset` | Reset cached clients | None |
| POST | `/v1/debug/test-openai` | Test OpenAI connection | None |
| GET | `/health` | Health check | None |

---

### POST `/v1/ingest` - Document Ingestion

**Description**: Upload one or more documents for processing and vector storage.

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File[] | Yes | One or more document files |
| `index` | string | No | Pinecone index name (defaults to `PINECONE_INDEX`) |
| `namespace` | string | No | Target namespace (required only for `routing_mode=manual`) |
| `routing_mode` | enum | No | `auto` (default), `manual`, or `per_chunk` |
| `metadata_json` | string | No | Custom JSON metadata to attach to document |

**Supported File Types**:
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.md` - Markdown files
- `.txt` - Plain text files
- `.json` - JSON files
- `.csv` - CSV files

**Response** (202 Accepted):

```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "document.pdf",
      "status": "queued"
    }
  ],
  "received_count": 1
}
```

**Error Responses**:

| Status | Condition |
|--------|-----------|
| 400 | Invalid filename, file type, namespace, or routing mode |
| 422 | Pydantic validation error |
| 500 | Server error |

---

### GET `/v1/ingest/{job_id}` - Job Status

**Description**: Retrieve the current status of an ingestion job.

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | UUID | Job identifier from ingestion response |

**Response** (200 OK):

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "status": "completed",
  "metadata": {
    "index": "personal-rag-index",
    "routing_mode": "auto",
    "namespace": null,
    "metadata": null
  },
  "file_path": "/tmp/rag-uploads/550e8400.../document.pdf",
  "error": null,
  "chunks_processed": 15,
  "routing_fallback_used": false
}
```

**Job Status Values**:

| Status | Description |
|--------|-------------|
| `queued` | Job created, waiting for processing slot |
| `chunking` | Parsing document with Docling |
| `routing` | Classifying chunks to namespaces |
| `embedding` | Generating embeddings with OpenAI |
| `upserting` | Storing vectors in Pinecone |
| `completed` | Successfully processed |
| `failed` | Processing failed (see `error` field) |

**Error Responses**:

| Status | Condition |
|--------|-----------|
| 400 | Invalid job ID format |
| 404 | Job not found |

---

### POST `/v1/debug/reset` - Reset Cached Clients

**Description**: Clear all cached clients and settings. Use after changing API keys.

**Response** (200 OK):

```json
{
  "status": "ok",
  "message": "All cached clients and settings cleared"
}
```

---

### POST `/v1/debug/test-openai` - Test OpenAI Connection

**Description**: Test OpenAI API connection with a minimal embedding request.

**Response** (200 OK - Success):

```json
{
  "status": "ok",
  "api_key_preview": "sk-proj-...",
  "embedding_dimensions": 1536
}
```

**Response** (200 OK - Error):

```json
{
  "status": "error",
  "api_key_preview": "sk-proj-...",
  "error_type": "AuthenticationError",
  "error_message": "Invalid API key"
}
```

---

### GET `/health` - Health Check

**Description**: Simple health check endpoint.

**Response** (200 OK):

```json
{
  "status": "ok"
}
```

---

## 6. Data Models

### Enums

#### RoutingMode

| Value | Description |
|-------|-------------|
| `auto` | LLM classifies entire document once (default) |
| `manual` | Use provided namespace, no LLM involved |
| `per_chunk` | LLM classifies each chunk individually |

#### Namespace

| Value | Description |
|-------|-------------|
| `personal_life` | Personal interests, hobbies, life events |
| `professional_life` | Work, career, professional achievements |
| `about_rag` | Information about the RAG system itself |

#### Job Status

| Value | Description |
|-------|-------------|
| `queued` | Waiting in queue |
| `chunking` | Parsing document |
| `routing` | Classifying to namespaces |
| `embedding` | Generating vectors |
| `upserting` | Storing in Pinecone |
| `completed` | Successfully finished |
| `failed` | Error occurred |

### Data Structures

#### ParsedChunk

Internal representation of a document chunk.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Chunk content |
| `metadata.source_path` | string | Original file path |
| `metadata.document_title` | string | Base filename |
| `metadata.heading` | string | Section heading |
| `metadata.page_number` | int | PDF page number |
| `metadata.chunk_index` | int | Position in document |
| `metadata.context_summary` | string | Hierarchical context |

#### Vector Schema (Pinecone)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | MD5 hash of chunk text |
| `values` | float[1536] | Embedding vector |
| `metadata.text` | string | Full chunk text |
| `metadata.contextualized_text` | string | Context summary |
| `metadata.doc_title` | string | Document title |
| `metadata.heading` | string | Section heading |
| `metadata.source_url` | string | Custom source URL |
| `metadata.page_number` | int | Page number |
| `metadata.content_type` | string | MIME type |
| `metadata.chunk_index` | int | Chunk position |

#### IngestJobRecord

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | UUID identifier |
| `filename` | string | Original filename |
| `content_type` | string | MIME content type |
| `status` | enum | Current processing status |
| `metadata` | object | Index, routing mode, custom metadata |
| `file_path` | string | Temporary file location |
| `error` | string | Error message if failed |
| `chunks_processed` | int | Number of chunks created |
| `routing_fallback_used` | bool | True if routing used fallback |

---

## 7. Document Processing Pipeline (ETL)

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT PROCESSING PIPELINE                         │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │  UPLOAD  │───▶│  PARSE   │───▶│  ROUTE   │───▶│  EMBED   │───▶│ UPSERT ││
│  │          │    │          │    │          │    │          │    │        ││
│  │ Save to  │    │ Docling  │    │OpenRouter│    │ OpenAI   │    │Pinecone││
│  │ temp dir │    │ Chunking │    │   LLM    │    │   API    │    │  gRPC  ││
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘│
│       │               │               │               │               │     │
│       ▼               ▼               ▼               ▼               ▼     │
│   "queued"       "chunking"      "routing"      "embedding"     "upserting" │
│                                                                              │
│                                                         ┌──────────────────┐│
│                                                         │    "completed"   ││
│                                                         │        or        ││
│                                                         │     "failed"     ││
│                                                         └──────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: File Upload & Queuing

**Location**: `app/api/ingest.py`, `app/services/file_storage.py`

**Process**:
1. Receive multipart form data with file(s)
2. Validate file extensions against whitelist
3. Validate routing mode and namespace compatibility
4. Parse optional metadata JSON
5. Save file to `/tmp/rag-uploads/{job_id}/{filename}`
6. Create `IngestJobRecord` with status `queued`
7. Add job to background task queue
8. Return 202 Accepted with job summaries

**File Storage**:
- Base directory: `/tmp/rag-uploads/`
- Job isolation: Each job gets unique subdirectory
- Cleanup: Only on successful completion

---

### Stage 2: Document Parsing (Chunking)

**Location**: `app/services/parser.py`

**Technology**: IBM Docling with HybridChunker

**Status**: `chunking`

**Process**:
1. Detect file type from extension
2. Route to appropriate parser:
   - `.pdf`, `.docx`, `.md` → Direct Docling parsing
   - `.txt`, `.json`, `.csv` → Normalize to Markdown, then Docling
3. Convert document with `DocumentConverter`
4. Chunk with `HybridChunker`:
   - Tokenizer: Configurable (default: gpt2)
   - Max tokens: 512 per chunk
   - Merge peers: Enabled
5. Extract metadata for each chunk:
   - Page number (from provenance)
   - Heading (first heading in hierarchy)
   - Document title (filename)
   - Chunk index (sequential)
6. Generate context summary via `chunker.contextualize()`
7. Return list of `ParsedChunk` objects

**Fallback Handling**:
- If HybridChunker returns no chunks → Use full markdown export
- If markdown is empty → Raise ValueError

**Text File Normalization**:
| Format | Normalization |
|--------|--------------|
| `.txt` | Read as UTF-8 |
| `.json` | Pretty-print with indent=2 |
| `.csv` | Convert via pandas to CSV string |

---

### Stage 3: Namespace Routing

**Location**: `app/services/namespace_router.py`

**Technology**: OpenRouter API with LLM

**Status**: `routing`

**Routing Modes**:

| Mode | Behavior |
|------|----------|
| `manual` | Use provided namespace for all chunks (no LLM) |
| `auto` | LLM classifies document once using first chunk + all headings |
| `per_chunk` | LLM classifies each chunk individually |

**Classification Process**:
1. Build classification prompt with:
   - Namespace descriptions
   - Document headings (if available)
   - First 2000 characters of content
2. POST to OpenRouter `/chat/completions`
3. Extract single-word response
4. Map to Namespace enum
5. Track if fallback was used

**LLM Configuration**:
- Model: `meta-llama/llama-3.2-3b-instruct:free`
- Temperature: 0 (deterministic)
- Timeout: 20 seconds
- Fallback: `PROFESSIONAL_LIFE`

**Error Handling**:
- Timeout → Log warning, use fallback, set `routing_fallback_used`
- HTTP error → Log with status code, use fallback
- Invalid response → Log warning, use fallback

---

### Stage 4: Embedding Generation

**Location**: `app/services/embedder.py`

**Technology**: OpenAI Embeddings API

**Status**: `embedding`

**Process**:
1. Collect contextualized text from all chunks
2. Batch texts into groups of 20
3. For each batch:
   - Call OpenAI embeddings API
   - Retry on rate limit (up to 10 times)
   - Wait 1 second between batches
4. Return flat list of embedding vectors

**Configuration**:
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Batch size: 20 texts
- Inter-batch delay: 1 second

**Retry Strategy**:
- Trigger: `RateLimitError`
- Max attempts: 10
- Wait: Exponential backoff with jitter
  - Initial: 5 seconds
  - Maximum: 120 seconds
  - Jitter: 5 seconds

---

### Stage 5: Vector Upserting

**Location**: `app/services/vectordb.py`

**Technology**: Pinecone gRPC Client

**Status**: `upserting`

**Process**:
1. Build vector objects with:
   - ID: MD5 hash of chunk text
   - Values: 1536-dim embedding
   - Metadata: Full chunk metadata
2. Group vectors by namespace
3. For each namespace:
   - Batch vectors into groups of 100
   - Upsert batch to Pinecone
   - Retry on connection errors
4. Return total upserted count

**Configuration**:
- Batch size: 100 vectors
- Protocol: gRPC

**Retry Strategy**:
- Trigger: `PineconeException`, `ConnectionError`
- Max attempts: 5
- Wait: Exponential backoff with jitter
  - Initial: 1 second
  - Maximum: 30 seconds

---

### Stage 6: Cleanup & Completion

**Location**: `app/services/ingest_queue.py`, `app/services/file_storage.py`

**Status**: `completed` or `failed`

**On Success**:
1. Set `chunks_processed` count
2. Update status to `completed`
3. Remove temporary files

**On Failure**:
1. Extract actual error from RetryError (if applicable)
2. Set `error` message with type and details
3. Update status to `failed`
4. Retain temporary files for debugging

---

## 8. External Integrations

### OpenAI Integration

**Purpose**: Document embedding generation

| Setting | Value |
|---------|-------|
| Endpoint | `https://api.openai.com/v1/embeddings` |
| Model | `text-embedding-3-small` |
| Output dimensions | 1536 |
| Authentication | Bearer token (API key) |
| Client library | `openai` Python SDK |

**Rate Limiting**:
- Batch size limited to 20 texts
- 1-second delay between batches
- Exponential backoff on 429 errors

**Debug Endpoint**: `POST /v1/debug/test-openai`

---

### OpenRouter Integration

**Purpose**: LLM-based namespace classification

| Setting | Value |
|---------|-------|
| Base URL | `https://openrouter.ai/api/v1` (configurable) |
| Endpoint | `POST /chat/completions` |
| Default model | `meta-llama/llama-3.2-3b-instruct:free` |
| Authentication | Bearer token (API key) |
| Client library | `httpx` |
| Timeout | 20 seconds |

**Request Format**:
```json
{
  "model": "meta-llama/llama-3.2-3b-instruct:free",
  "messages": [{"role": "user", "content": "<classification prompt>"}],
  "temperature": 0,
  "stream": false
}
```

**Model Selection**:
- Automatically appends `:free` suffix if not present
- Uses free-tier models to minimize costs

---

### Pinecone Integration

**Purpose**: Vector storage and semantic search

| Setting | Value |
|---------|-------|
| Protocol | gRPC |
| Client library | `pinecone` Python SDK (PineconeGRPC) |
| Batch size | 100 vectors per upsert |
| Host resolution | Per-index or global override |

**Host Configuration**:
1. Check `PINECONE_HOST_{INDEX_NAME}` (uppercase, underscores)
2. Fall back to `PINECONE_HOST` (global)
3. Error if neither configured

**Namespace Organization**:
- `personal_life` - Personal content
- `professional_life` - Professional content
- `about_rag` - System documentation

---

### Docling Integration

**Purpose**: Advanced document parsing and OCR

| Setting | Value |
|---------|-------|
| Converter | `DocumentConverter` |
| Chunker | `HybridChunker` |
| Tokenizer | Configurable (default: gpt2) |
| Max tokens | 512 per chunk |
| Merge peers | Enabled |

**Capabilities**:
- PDF text extraction with layout preservation
- OCR for scanned documents
- Table structure detection
- Heading hierarchy extraction
- Multi-format support (PDF, DOCX, Markdown)

**Contextualization**:
- `chunker.contextualize(chunk)` provides hierarchical context
- Includes document title and heading path
- Used for improved embedding quality

---

## 9. Namespace Routing System

### Namespace Definitions

| Namespace | Purpose | Example Content |
|-----------|---------|-----------------|
| `personal_life` | Personal interests, hobbies, life events | Travel journals, personal projects, family |
| `professional_life` | Work, career, achievements | Resumes, work projects, certifications |
| `about_rag` | RAG system documentation | Architecture docs, API specs, guides |

### Routing Modes Comparison

| Mode | LLM Calls | Use Case |
|------|-----------|----------|
| `manual` | 0 | Known category, bulk imports |
| `auto` | 1 | Mixed documents, single category |
| `per_chunk` | N (one per chunk) | Multi-topic documents |

### Classification Prompt Structure

```
You are a classifier for a personal portfolio RAG system...

<namespace descriptions>

Analyze the following content and respond with ONLY the namespace name...

Document headings:
- <heading 1>
- <heading 2>

<first 2000 characters of content>
```

### Fallback Behavior

| Condition | Action | Flag Set |
|-----------|--------|----------|
| Empty content | Return `PROFESSIONAL_LIFE` | Yes |
| Missing API key | Return `PROFESSIONAL_LIFE` | Yes |
| Timeout | Return `PROFESSIONAL_LIFE` | Yes |
| HTTP error | Return `PROFESSIONAL_LIFE` | Yes |
| Invalid response | Return `PROFESSIONAL_LIFE` | Yes |
| Unrecognized namespace | Return `PROFESSIONAL_LIFE` | Yes |

**Tracking**: `routing_fallback_used` field in job record indicates if fallback was used

---

## 10. Vector Storage & Retrieval

### Vector Schema

Each vector stored in Pinecone contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | MD5 hash of chunk text |
| `values` | float[1536] | OpenAI embedding vector |
| `metadata.text` | string | Original chunk text |
| `metadata.contextualized_text` | string | Hierarchical context summary |
| `metadata.doc_title` | string | Source document title |
| `metadata.heading` | string | Section heading |
| `metadata.source_url` | string | Custom source URL (user-provided) |
| `metadata.page_number` | int | PDF page number (1-indexed) |
| `metadata.content_type` | string | MIME type |
| `metadata.chunk_index` | int | Position in document (1-indexed) |

### ID Generation

- Algorithm: MD5 hash of chunk text
- Format: 32-character hexadecimal string
- Collision handling: Overwrites existing vector with same ID

### Namespace Strategy

- Vectors grouped by semantic category
- Enables filtered/targeted retrieval
- Supports multi-tenant scenarios

### Batch Processing

- Upsert batch size: 100 vectors
- Automatic batching for large documents
- Progress logging for each batch

---

## 11. Concurrency & Resource Management

### Job Processing Semaphore

**Purpose**: Prevent resource exhaustion from parallel document processing

**Implementation**:
- `asyncio.Semaphore(1)` - Single concurrent job
- Jobs wait for available slot
- Acquired before processing, released after

**Why Limit to 1**:
- Docling is memory-intensive
- OpenAI API has rate limits
- Prevents OOM in container environments
- Ensures predictable processing times

### Thread Safety

**Protected Resources**:
- `JOB_STORE` - Dict of job records
- `JOB_QUEUE` - List of pending job IDs

**Mechanism**:
- `threading.Lock` for all access
- Atomic status updates via `_update_status()`
- Safe concurrent reads during status polling

### Async Architecture

```
Request Thread                    Background Task
      │                                 │
      ▼                                 │
  Validate                              │
      │                                 │
      ▼                                 │
  Save File                             │
      │                                 │
      ▼                                 │
  Add to Queue                          │
      │                                 │
      ▼                                 │
  Return 202 ─────────────────────▶ Acquire Semaphore
                                        │
                                        ▼
                                  Run in Executor
                                        │
                                        ▼
                                  process_job()
                                        │
                                        ▼
                                  Release Semaphore
```

---

## 12. Error Handling & Retry Logic

### Retry Configurations

| Service | Exception Types | Max Attempts | Initial Wait | Max Wait |
|---------|-----------------|--------------|--------------|----------|
| Embedder | `RateLimitError` | 10 | 5s | 120s |
| VectorDB | `PineconeException`, `ConnectionError` | 5 | 1s | 30s |

### Retry Strategy Details

**Exponential Backoff with Jitter**:
- Wait time increases exponentially between attempts
- Random jitter prevents thundering herd
- Capped at maximum wait time

**tenacity Configuration**:
```python
@retry(
    retry=retry_if_exception_type(ExceptionClass),
    stop=stop_after_attempt(N),
    wait=wait_exponential_jitter(initial=X, max=Y),
    reraise=True,
)
```

### Error Propagation

1. Exception raised in processing stage
2. Caught in `process_job()` outer try/except
3. RetryError unwrapped to get actual cause
4. Error message formatted: `{ErrorType}: {message}`
5. Stored in job record `error` field
6. Status set to `failed`
7. Logged with job context

### Validation Errors

| Stage | Error Type | HTTP Status |
|-------|------------|-------------|
| File type | ValueError | 400 |
| Namespace | ValueError | 400 |
| Routing mode | ValueError | 400 |
| Index | ValueError | 400 |
| Job ID format | ValueError | 400 |
| Metadata JSON | ValueError | 400 |
| Job not found | N/A | 404 |

---

## 13. Logging & Observability

### Logging Configuration

**Setup**: `app/core/logging.py`

**Format**:
```
%(asctime)s | %(levelname)s | %(name)s | %(message)s
```

**Example**:
```
2024-01-15 10:30:45,123 | INFO | app.services.ingest_queue | Job abc123: starting processing
```

**Handler**: StreamHandler to stdout

**Default Level**: INFO (set on startup)

### Log Points

| Stage | Level | Message Pattern |
|-------|-------|-----------------|
| Job start | INFO | `Job {id}: starting processing` |
| Job config | INFO | `Job {id}: file={name}, index={idx}, routing_mode={mode}` |
| Parsing start | INFO | `Job {id}: parsing file` |
| Parsing complete | INFO | `Job {id}: parsed {n} chunks` |
| Routing start | INFO | `Job {id}: routing chunks (mode={mode})` |
| Fallback used | WARN | `Job {id}: routing used fallback namespace` |
| Routing complete | INFO | `Job {id}: routed to namespaces {set}` |
| Embedding start | INFO | `Job {id}: embedding {n} chunks` |
| Embedding batch | DEBUG | `Embedding batch {i}/{n} ({count} texts)` |
| Embedding complete | INFO | `Job {id}: embedded {n} chunks` |
| Upsert start | INFO | `Job {id}: upserting to {n} namespaces` |
| Upsert batch | DEBUG | `Upserted batch {i}/{n} ({count} vectors)` |
| Job complete | INFO | `Job {id}: completed successfully, {n} chunks processed` |
| Job failed | ERROR | `Job {id}: failed - {ErrorType}: {message}` |

### Structured Logging Fields

- `job_id` - Always included for job-related logs
- `filename` - Document being processed
- `index` - Target Pinecone index
- `routing_mode` - Selected routing mode
- `chunks` - Number of chunks
- `namespaces` - Target namespace(s)

---

## 14. Docker & Deployment

### Production Dockerfile

**File**: `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Characteristics**:
- Minimal base image
- Only essential system dependencies
- No development tools
- Single-stage build
- Optimized with opencv-python-headless

---

### Development Dockerfile

**File**: `Dockerfile.dev`

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000",
     "--reload", "--reload-dir", "/app/app"]
```

**Characteristics**:
- Hot reload enabled
- Code mounted from host
- Watches `/app/app` for changes

---

### Docker Compose (Development)

**File**: `docker-compose.dev.yml`

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    env_file:
      - my.env
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
```

**Usage**:
```bash
# Build and start
docker-compose -f docker-compose.dev.yml up --build

# Rebuild after dependency changes
docker-compose -f docker-compose.dev.yml build --no-cache

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop
docker-compose -f docker-compose.dev.yml down
```

---

### Deployment Targets

| Environment | Platform | Configuration |
|-------------|----------|---------------|
| Development | Local Docker | docker-compose.dev.yml |
| Production | Railway | Dockerfile (auto-detected) |
| Frontend | Vercel | Separate repository |

---

## 15. Security Considerations

### Input Validation

| Check | Implementation |
|-------|----------------|
| File extension whitelist | `.pdf`, `.docx`, `.md`, `.txt`, `.json`, `.csv` |
| Filename required | Non-empty string |
| Namespace validation | Enum membership |
| Routing mode validation | Enum membership |
| Index required | Non-empty string |
| Job ID format | Valid UUID |
| Metadata JSON | Well-formed JSON object |

### API Key Security

| Practice | Implementation |
|----------|----------------|
| Key storage | Environment variables only |
| Key masking | First 8 chars + "..." in responses |
| No key logging | Keys never appear in logs |
| Key rotation | `POST /v1/debug/reset` clears cached clients |

### File Handling

| Practice | Implementation |
|----------|----------------|
| Upload isolation | Job-specific subdirectory |
| Temporary storage | `/tmp/rag-uploads/` |
| Path handling | Absolute paths only |
| Cleanup policy | Remove on success, retain on failure |
| Size limits | Inherited from FastAPI/Uvicorn defaults |

### Content Validation

| Check | Implementation |
|-------|----------------|
| Empty file detection | Raise error if no content extracted |
| Malformed document | Docling error propagation |
| Encoding | UTF-8 enforced for text files |

---

## 16. System Limitations & Configurations

### Processing Limits

| Parameter | Value | Configurable |
|-----------|-------|--------------|
| Max concurrent jobs | 1 | Code change required |
| Embedding batch size | 20 | `EMBEDDING_BATCH_SIZE` constant |
| Inter-batch delay | 1 second | `INTER_BATCH_DELAY` constant |
| Upsert batch size | 100 | `UPSERT_BATCH_SIZE` constant |
| Chunk size | ~512 tokens | HybridChunker config |
| Classification prompt limit | 2000 chars | Code change required |
| OpenRouter timeout | 20 seconds | Code change required |

### Vector Specifications

| Parameter | Value |
|-----------|-------|
| Embedding model | text-embedding-3-small |
| Vector dimensions | 1536 |
| ID generation | MD5 hash of chunk text |
| Metadata size | Limited by Pinecone tier |

### Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single concurrent job | Sequential processing | Scale horizontally with multiple instances |
| In-memory job store | Lost on restart | Implement persistent storage |
| No authentication | Public API | Add API key middleware |
| No rate limiting | Vulnerable to abuse | Add rate limiting middleware |
| No file size limits | Memory issues | Configure Uvicorn limits |
| MD5 ID collision | Overwrites vectors | Use content-aware hashing |

### Recommended Production Improvements

1. **Persistent Job Store**: Redis or PostgreSQL for job state
2. **Authentication**: API key or OAuth2 middleware
3. **Rate Limiting**: Redis-based rate limiting
4. **File Size Limits**: Nginx or Uvicorn configuration
5. **Horizontal Scaling**: Multiple instances with shared queue
6. **Monitoring**: Prometheus metrics, Grafana dashboards
7. **Alerting**: Error rate and latency alerts

---

## Appendix: Quick Reference

### Supported File Types

| Extension | MIME Type | Parser |
|-----------|-----------|--------|
| `.pdf` | application/pdf | Docling (direct) |
| `.docx` | application/vnd.openxmlformats-officedocument.wordprocessingml.document | Docling (direct) |
| `.md` | text/markdown | Docling (direct) |
| `.txt` | text/plain | Normalized → Docling |
| `.json` | application/json | Normalized → Docling |
| `.csv` | text/csv | Normalized → Docling |

### Environment Variable Checklist

```bash
# Required
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=your-index-name

# Optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free
PINECONE_HOST=https://your-index.svc.region.pinecone.io
DOCLING_TOKENIZER=gpt2
```

### Common curl Commands

```bash
# Health check
curl http://localhost:8000/health

# Upload document (auto routing)
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@document.pdf" \
  -F "index=my-index"

# Upload with manual namespace
curl -X POST http://localhost:8000/v1/ingest \
  -F "file=@document.pdf" \
  -F "index=my-index" \
  -F "namespace=personal_life" \
  -F "routing_mode=manual"

# Check job status
curl http://localhost:8000/v1/ingest/{job_id}

# Test OpenAI connection
curl -X POST http://localhost:8000/v1/debug/test-openai

# Reset cached clients
curl -X POST http://localhost:8000/v1/debug/reset
```

---

*Document generated: February 2025*
*System version: 1.0*
*Total codebase: ~1,600 lines of Python*

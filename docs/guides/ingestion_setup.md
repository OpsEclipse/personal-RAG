# Ingestion Setup Guide

This guide explains how to configure and run the FastAPI ingestion service for Docling + OpenAI embeddings + Pinecone.

## Prerequisites

- Python 3.11+
- A Pinecone account and index
- An OpenAI API key
- Access to document sources (URLs or server-accessible file paths)

## Environment Variables

Create a `my.env` file in the project root and populate the following:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-index
```


## Install Dependencies

```
pip install -r requirements.txt
```

## Run the API

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Ingest Documents

Send a POST request to the ingestion endpoint with one or more file uploads.

```
POST /v1/ingest (multipart/form-data)
- file: handbook.pdf
- file: policies.docx
- routing_mode: manual | auto | per_chunk
- namespace: about_rag (required only when routing_mode=manual; must be omitted otherwise)
- index: rag-index (optional; defaults to PINECONE_INDEX)
- metadata_json: {"document_title":"Company Handbook","source_system":"upload"}
```

## Notes

- Ingestion is asynchronous; the API returns `202 Accepted` with a job list (one per file).
- File paths must be accessible to the API server at runtime.
- The Pinecone namespace and index determine where vectors are stored when routing_mode=manual.
- If `index` is omitted, the API uses `PINECONE_INDEX` from `my.env`.
- Valid namespaces: personal_life, professional_life, about_rag.
- To bypass LLM routing, set routing_mode=manual and provide namespace.
- When routing_mode is auto or per_chunk, namespace must be omitted.

## Check Job Status

Use each job id from the ingestion response to check status:

```
GET /v1/ingest/{job_id}
```
Optional:

```
PINECONE_NAMESPACE=default
```

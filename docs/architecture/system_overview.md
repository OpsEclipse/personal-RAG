# System Overview & Architecture

The flow is divided into three primary phases: **Request Ingress**, **Document Processing (ETL)**, and **Vector Storage**.

## 1. Request Ingress & Backend Orchestration

- **Frontend:** A user interface hosted on **Vercel** initiates the process via an **API call**.
- **Backend:** A **FastAPI** server hosted on **Railway** receives the request.
- **Asynchronous Queue:** To handle heavy document processing without blocking the API, data is sent to a **Queue**. This ensures the system remains scalable and resilient under load.

## 2. Advanced Document Processing (The "Docling" Pipeline)

The system uses **Docling** (likely IBM's Docling) to transform raw files into structured data.

- **OCR & Extraction:** The data undergoes OCR to extract text while strictly preserving structural elements like **tables and formatting**.
- **Intelligent Chunking:** Instead of simple character splitting, it uses a `DoclingLoader` with a `HybridChunker` for optimal document splitting based on the document's actual layout.
- **Metadata Formatting:** A formatting script enriches each chunk with vital metadata, including:
  - Page numbers and header paths.
  - Source URLs and document titles.
- **Contextualization:** Before embedding, the system runs a `.contextualize()` function on each chunk. This adds a `context_summary` to the metadata, which helps the LLM understand how a specific chunk fits into the broader document.

## 3. Embedding & Vector Storage

- **Model:** The system utilizes the **OpenAI `text-embedding-3-small`** model, generating vectors with **1,536 dimensions**.
- **Final Chunk Schema:** The processed data is structured into a JSON-like object containing:
  - `id`: MD5 hash of the text + heading.
  - `values`: The 1536-dimension vector.
  - `metadata`: Comprehensive info including text, headings, page numbers, and the context summary.
- **Vector Database:** The vectors are upserted into **Pinecone**.
- **Retrieval Strategy:**
  - **Namespacing:** Data is inserted into specific namespaces/indexes for organization.
  - **Semantic Routing:** The system is designed to use semantic routing for namespace searches, ensuring the retrieval step is targeted and efficient.
  - **Re-ranking:** The plan includes a re-ranker step that retrieves the top 20 candidates and narrows them down to the **top 5** for the final LLM prompt.

## Technical Stack Summary

| Component | Technology |
| --- | --- |
| **Frontend** | Vercel |
| **Backend API** | FastAPI (Railway) |
| **Parsing/OCR** | Docling (with HybridChunker) |
| **Embedding Model** | OpenAI `text-embedding-3-small` |
| **Vector Store** | Pinecone |
| **Orchestration** | Queue-based data flow |

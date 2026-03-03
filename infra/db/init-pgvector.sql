-- Enable pgvector for RAG embedding storage. Run once per database (or use app's init_db() which does this).
CREATE EXTENSION IF NOT EXISTS vector;

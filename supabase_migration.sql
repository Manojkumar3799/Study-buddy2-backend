-- Run this ONCE in Supabase Dashboard -> SQL Editor -> New query -> Run.
-- This is the missing piece: your backend code (vector_store_service.py,
-- retrieval_service.py) assumes this table already exists. It doesn't yet.

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Table that stores every chunk + its embedding, per document
create table if not exists document_chunks (
    id          bigserial primary key,
    document_id text not null,
    chunk_id    integer not null,
    text        text not null,
    start_page  integer not null,
    end_page    integer not null,
    -- 768 dims to match gemini-embedding-001 (see embedding_service.py: dimensions=768)
    embedding   vector(768) not null,
    created_at  timestamptz not null default now()
);

-- 3. Index for fast lookup by document_id (used on every store/retrieve/delete)
create index if not exists idx_document_chunks_document_id
    on document_chunks (document_id);

-- 4. Index for fast cosine-similarity search (used on every /ask call)
create index if not exists idx_document_chunks_embedding
    on document_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Rockaway Deal Intelligence - Supabase schema for analysis persistence
-- Run via: supabase db push (or apply manually in Supabase SQL Editor)

-- Companies: extracted/specter company metadata
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    industry TEXT,
    tagline TEXT,
    about TEXT,
    team JSONB,
    domain TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Jobs: one per upload/analysis run (ties analyses together)
-- job_id_legacy: short string from app (e.g. first 8 chars of UUID) for URL stability
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id_legacy TEXT UNIQUE,
    input_mode TEXT NOT NULL DEFAULT 'pitchdeck',
    vc_investment_strategy TEXT,
    instructions TEXT,
    use_web_search BOOLEAN NOT NULL DEFAULT false,
    run_config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Pitch decks: uploaded documents per company (storage_path points to Supabase Storage object)
CREATE TABLE IF NOT EXISTS pitch_decks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pitch_decks_company ON pitch_decks(company_id);

-- Chunks: evidence chunks for Evidence sheet and provenance
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pitch_deck_id UUID NOT NULL REFERENCES pitch_decks(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    source_file TEXT NOT NULL,
    page_or_slide TEXT,
    sort_order INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_pitch_deck ON chunks(pitch_deck_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_pitch_deck_chunk ON chunks(pitch_deck_id, chunk_id);

-- Analyses: full pipeline state per company/pitch deck
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pitch_deck_id UUID REFERENCES pitch_decks(id) ON DELETE SET NULL,
    company_id UUID REFERENCES companies(id) ON DELETE SET NULL,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    state JSONB NOT NULL DEFAULT '{}',
    results_payload JSONB,
    status TEXT NOT NULL DEFAULT 'done',
    error TEXT,
    run_config JSONB DEFAULT '{}',
    excel_storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyses_job ON analyses(job_id);
CREATE INDEX IF NOT EXISTS idx_analyses_job_legacy ON analyses(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_analyses_company ON analyses(company_id);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);

-- RLS disabled for MVP; app uses password protection. Use service_role key server-side.

-- Shared persisted company chat sessions
-- Stores transcript history, selected answer model, costs, and raw web-search citation data.

CREATE TABLE IF NOT EXISTS company_chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_key TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    selection JSONB NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',
    transcript JSONB NOT NULL DEFAULT '[]',
    model_executions JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_company_chat_sessions_updated_at
    ON company_chat_sessions(updated_at DESC);

ALTER TABLE company_chat_sessions ENABLE ROW LEVEL SECURITY;

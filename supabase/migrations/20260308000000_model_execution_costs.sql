-- Run-level LLM and Perplexity cost telemetry.

ALTER TABLE model_executions
    ADD COLUMN IF NOT EXISTS service TEXT NOT NULL DEFAULT 'llm',
    ADD COLUMN IF NOT EXISTS estimated_cost_usd DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS request_count INT;

CREATE INDEX IF NOT EXISTS idx_model_executions_service
    ON model_executions(service);

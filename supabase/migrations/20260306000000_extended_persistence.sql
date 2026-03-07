-- Extended persistence for Rockaway Deal Intelligence
-- Adds: analysis_events, job_controls, job_status_history, analysis_errors,
--       source_files, model_executions, person_profile_jobs, and job version fields.

ALTER TABLE jobs
    ADD COLUMN IF NOT EXISTS app_version TEXT,
    ADD COLUMN IF NOT EXISTS prompt_version TEXT,
    ADD COLUMN IF NOT EXISTS pipeline_version TEXT,
    ADD COLUMN IF NOT EXISTS schema_version TEXT;

CREATE TABLE IF NOT EXISTS analysis_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'progress',
    stage TEXT,
    message TEXT NOT NULL,
    payload JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_events_job_legacy ON analysis_events(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_analysis_events_created_at ON analysis_events(created_at DESC);

CREATE TABLE IF NOT EXISTS job_controls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL UNIQUE,
    pause_requested BOOLEAN NOT NULL DEFAULT false,
    stop_requested BOOLEAN NOT NULL DEFAULT false,
    last_action TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_job_controls_job_legacy ON job_controls(job_id_legacy);

CREATE TABLE IF NOT EXISTS job_status_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    status TEXT NOT NULL,
    progress TEXT,
    source TEXT NOT NULL DEFAULT 'app',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_job_status_history_job_legacy ON job_status_history(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_job_status_history_created_at ON job_status_history(created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    company_slug TEXT,
    stage TEXT,
    error_type TEXT,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_errors_job_legacy ON analysis_errors(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_analysis_errors_created_at ON analysis_errors(created_at DESC);

CREATE TABLE IF NOT EXISTS source_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size_bytes BIGINT,
    mime_type TEXT,
    sha256 TEXT,
    local_path TEXT,
    storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_source_files_job_legacy ON source_files(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_source_files_sha256 ON source_files(sha256);

CREATE TABLE IF NOT EXISTS model_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    job_id_legacy TEXT NOT NULL,
    company_slug TEXT,
    stage TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    request_timeout_seconds DOUBLE PRECISION,
    max_retries INT,
    latency_ms INT,
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    status TEXT NOT NULL DEFAULT 'done',
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_executions_job_legacy ON model_executions(job_id_legacy);
CREATE INDEX IF NOT EXISTS idx_model_executions_created_at ON model_executions(created_at DESC);

CREATE TABLE IF NOT EXISTS person_profile_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_job_id TEXT NOT NULL UNIQUE,
    company_slug TEXT,
    person_key TEXT,
    status TEXT NOT NULL,
    progress TEXT,
    request_payload JSONB DEFAULT '{}',
    result_payload JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_person_profile_jobs_status ON person_profile_jobs(status);
CREATE INDEX IF NOT EXISTS idx_person_profile_jobs_company_slug ON person_profile_jobs(company_slug);

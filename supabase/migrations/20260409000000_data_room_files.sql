-- Data Room: file-sharing between startups and matched VCs
-- Startups upload files to the data room; matched VCs can browse and download.
-- Files may optionally also be ingested as pipeline evidence (also_evidence flag).

CREATE TABLE IF NOT EXISTS data_room_files (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id    UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
  storage_path  TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_size_bytes BIGINT,
  mime_type     TEXT,
  category      TEXT NOT NULL DEFAULT 'other',  -- pitch_deck, financials, legal, team, product, other
  also_evidence BOOLEAN NOT NULL DEFAULT false,
  pitch_deck_id UUID REFERENCES pitch_decks(id) ON DELETE SET NULL,
  uploaded_by   UUID REFERENCES users_profile(id),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_data_room_files_company
  ON data_room_files(company_id);

CREATE INDEX IF NOT EXISTS idx_data_room_files_company_created
  ON data_room_files(company_id, created_at DESC);

ALTER TABLE data_room_files ENABLE ROW LEVEL SECURITY;

-- Allow service-role full access (used by backend)
CREATE POLICY "Service role full access on data_room_files"
  ON data_room_files FOR ALL
  USING (true)
  WITH CHECK (true);

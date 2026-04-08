-- Sprint 3: debates + debate_messages tables with RLS
-- Run in Supabase SQL editor before deploying Sprint 3 code.

-- ============================================================
-- debates: one debate per (match, startup) pair
-- ============================================================

CREATE TABLE IF NOT EXISTS debates (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  match_id        UUID        NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
  company_id      UUID        NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
  vc_profile_id   UUID        NOT NULL REFERENCES vc_profiles(id) ON DELETE CASCADE,
  status          TEXT        NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'paused', 'completed')),
  current_round   INTEGER     NOT NULL DEFAULT 1,
  max_rounds      INTEGER     NOT NULL DEFAULT 3,
  summary         TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_debates_match_id ON debates(match_id);
CREATE INDEX IF NOT EXISTS idx_debates_company_id ON debates(company_id);
CREATE INDEX IF NOT EXISTS idx_debates_vc_profile_id ON debates(vc_profile_id);

ALTER TABLE debates ENABLE ROW LEVEL SECURITY;

-- VCs can read debates for their matches
CREATE POLICY "VCs see own debates"
  ON debates FOR SELECT
  USING (
    vc_profile_id IN (SELECT id FROM vc_profiles WHERE user_id = auth.uid())
  );

-- Startups can read debates for their company
CREATE POLICY "Startups see their company debates"
  ON debates FOR SELECT
  USING (
    company_id IN (
      SELECT company_id FROM user_company_links WHERE user_id = auth.uid()
    )
  );

-- ============================================================
-- debate_messages: one row per agent turn
-- ============================================================

CREATE TABLE IF NOT EXISTS debate_messages (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  debate_id   UUID        NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
  round       INTEGER     NOT NULL,
  speaker     TEXT        NOT NULL CHECK (speaker IN ('vc_agent', 'startup_agent', 'system')),
  content     TEXT        NOT NULL,
  citations   JSONB,        -- list of {chunk_id, excerpt, source_file}
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_debate_messages_debate_id ON debate_messages(debate_id);
CREATE INDEX IF NOT EXISTS idx_debate_messages_debate_round ON debate_messages(debate_id, round);

ALTER TABLE debate_messages ENABLE ROW LEVEL SECURITY;

-- Access via parent debate
CREATE POLICY "VCs see own debate messages"
  ON debate_messages FOR SELECT
  USING (
    debate_id IN (
      SELECT id FROM debates
      WHERE vc_profile_id IN (SELECT id FROM vc_profiles WHERE user_id = auth.uid())
    )
  );

CREATE POLICY "Startups see their debate messages"
  ON debate_messages FOR SELECT
  USING (
    debate_id IN (
      SELECT id FROM debates
      WHERE company_id IN (
        SELECT company_id FROM user_company_links WHERE user_id = auth.uid()
      )
    )
  );

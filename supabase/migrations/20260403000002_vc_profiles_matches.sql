-- Sprint 2: vc_profiles + matches tables with RLS
-- Run in Supabase SQL editor before deploying Sprint 2 code.

-- ============================================================
-- vc_profiles: one row per VC user, holds thesis & thresholds
-- ============================================================

CREATE TABLE IF NOT EXISTS vc_profiles (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          UUID        NOT NULL REFERENCES users_profile(id) ON DELETE CASCADE,
  firm_name        TEXT        NOT NULL,
  investment_thesis TEXT,
  min_strategy_fit INTEGER     NOT NULL DEFAULT 0 CHECK (min_strategy_fit >= 0 AND min_strategy_fit <= 100),
  min_team         INTEGER     NOT NULL DEFAULT 0 CHECK (min_team >= 0 AND min_team <= 100),
  min_potential    INTEGER     NOT NULL DEFAULT 0 CHECK (min_potential >= 0 AND min_potential <= 100),
  active           BOOLEAN     NOT NULL DEFAULT true,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One profile per VC user
CREATE UNIQUE INDEX IF NOT EXISTS idx_vc_profiles_user_id ON vc_profiles(user_id);
-- Fast lookup of active profiles during matching
CREATE INDEX IF NOT EXISTS idx_vc_profiles_active ON vc_profiles(active) WHERE active = true;

ALTER TABLE vc_profiles ENABLE ROW LEVEL SECURITY;

-- VCs can read/write their own profile; service-role bypasses RLS for matching
CREATE POLICY "VCs manage own profile"
  ON vc_profiles FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- ============================================================
-- matches: one row per (vc_profile, company) pair
-- ============================================================

CREATE TABLE IF NOT EXISTS matches (
  id               UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  vc_profile_id    UUID    NOT NULL REFERENCES vc_profiles(id) ON DELETE CASCADE,
  company_id       UUID    NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
  analysis_id      UUID    REFERENCES analyses(id),
  strategy_fit_score FLOAT,
  team_score       FLOAT,
  potential_score  FLOAT,
  composite_score  FLOAT,
  bucket           TEXT,
  status           TEXT    NOT NULL DEFAULT 'new'
                   CHECK (status IN ('new', 'viewed', 'interested', 'passed', 'in_debate')),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Each (vc, company) pair appears at most once
CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_vc_company ON matches(vc_profile_id, company_id);
CREATE INDEX IF NOT EXISTS idx_matches_vc_profile_id ON matches(vc_profile_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);

ALTER TABLE matches ENABLE ROW LEVEL SECURITY;

-- VCs can read their own matches and update status
CREATE POLICY "VCs see own matches"
  ON matches FOR SELECT
  USING (
    vc_profile_id IN (SELECT id FROM vc_profiles WHERE user_id = auth.uid())
  );

CREATE POLICY "VCs update own match status"
  ON matches FOR UPDATE
  USING (
    vc_profile_id IN (SELECT id FROM vc_profiles WHERE user_id = auth.uid())
  )
  WITH CHECK (
    vc_profile_id IN (SELECT id FROM vc_profiles WHERE user_id = auth.uid())
  );

-- Startups can see matches for their linked companies
CREATE POLICY "Startups see their company matches"
  ON matches FOR SELECT
  USING (
    company_id IN (
      SELECT company_id FROM user_company_links WHERE user_id = auth.uid()
    )
  );

-- Sprint 1: Fundraising flag and profile claiming metadata on companies.

ALTER TABLE companies
    ADD COLUMN IF NOT EXISTS fundraising BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS fundraising_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS data_room_enabled BOOLEAN NOT NULL DEFAULT false;

-- Index for matching engine queries (Sprint 2): quickly find companies that are fundraising.
CREATE INDEX IF NOT EXISTS idx_companies_fundraising
    ON companies(fundraising)
    WHERE fundraising = true;

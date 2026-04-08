-- Sprint 1: Per-user identity and startup profile claiming.
-- users_profile links to auth.users and carries the role (admin/vc/startup).
-- user_company_links associates startup users with their claimed company.

CREATE TABLE users_profile (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'vc', 'startup')),
    display_name TEXT,
    organization TEXT,
    approved BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE users_profile ENABLE ROW LEVEL SECURITY;

-- Users can read and update their own profile row.
CREATE POLICY "Users see own profile"
    ON users_profile FOR ALL
    USING (auth.uid() = id);

-- Admins can see all profiles.
CREATE POLICY "Admins see all profiles"
    ON users_profile FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM users_profile up
            WHERE up.id = auth.uid() AND up.role = 'admin'
        )
    );


CREATE TABLE user_company_links (
    user_id UUID NOT NULL REFERENCES users_profile(id) ON DELETE CASCADE,
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    role_in_company TEXT,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, company_id)
);

ALTER TABLE user_company_links ENABLE ROW LEVEL SECURITY;

-- Users can read and manage their own company links.
CREATE POLICY "Users see own company links"
    ON user_company_links FOR ALL
    USING (auth.uid() = user_id);

-- Admins can see all links.
CREATE POLICY "Admins see all company links"
    ON user_company_links FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM users_profile up
            WHERE up.id = auth.uid() AND up.role = 'admin'
        )
    );

-- Index for joining on company_id (e.g. "who claimed this company?").
CREATE INDEX idx_user_company_links_company_id
    ON user_company_links(company_id);

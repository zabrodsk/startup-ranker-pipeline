-- Debate evidence-request loop: structured pause/resume when the VC agent needs
-- more information from the founder. Extends debate_messages with a typed channel
-- and adds a simple lock column on debates so the respond endpoint can atomically
-- claim the founder's response.
--
-- Also defensively flips debates.match_id to ON DELETE SET NULL so a stray match
-- deletion can never destroy a debate via cascade again. The application layer
-- (web/app.py _run_re_evaluation) has also been updated to stop deleting matches
-- during re-evaluation; this FK change is belt-and-braces.

-- ============================================================
-- 1. debate_messages: new message_type channel + typed payloads
-- ============================================================

ALTER TABLE debate_messages
  ADD COLUMN IF NOT EXISTS message_type TEXT NOT NULL DEFAULT 'argument';

-- The CHECK is added as a separate statement so re-running the migration
-- against a DB that already has the column is idempotent.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'debate_messages'
      AND constraint_name = 'debate_messages_message_type_check'
  ) THEN
    ALTER TABLE debate_messages
      ADD CONSTRAINT debate_messages_message_type_check
      CHECK (message_type IN ('argument','evidence_request','founder_response','system_note'));
  END IF;
END $$;

ALTER TABLE debate_messages
  ADD COLUMN IF NOT EXISTS info_request JSONB;

ALTER TABLE debate_messages
  ADD COLUMN IF NOT EXISTS founder_response_type TEXT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'debate_messages'
      AND constraint_name = 'debate_messages_founder_response_type_check'
  ) THEN
    ALTER TABLE debate_messages
      ADD CONSTRAINT debate_messages_founder_response_type_check
      CHECK (founder_response_type IS NULL OR founder_response_type IN ('uploaded','unavailable'));
  END IF;
END $$;

ALTER TABLE debate_messages
  ADD COLUMN IF NOT EXISTS linked_reeval_job_id TEXT;

-- ============================================================
-- 2. debates: awaiting_input_from lock column
-- ============================================================

ALTER TABLE debates
  ADD COLUMN IF NOT EXISTS awaiting_input_from TEXT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'debates'
      AND constraint_name = 'debates_awaiting_input_from_check'
  ) THEN
    ALTER TABLE debates
      ADD CONSTRAINT debates_awaiting_input_from_check
      CHECK (awaiting_input_from IS NULL OR awaiting_input_from IN ('founder','vc'));
  END IF;
END $$;

-- ============================================================
-- 3. Defensively flip debates.match_id FK to ON DELETE SET NULL
-- ============================================================

ALTER TABLE debates ALTER COLUMN match_id DROP NOT NULL;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'debates'
      AND constraint_name = 'debates_match_id_fkey'
  ) THEN
    ALTER TABLE debates DROP CONSTRAINT debates_match_id_fkey;
  END IF;
END $$;

ALTER TABLE debates
  ADD CONSTRAINT debates_match_id_fkey
  FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE SET NULL;

-- ============================================================
-- 4. Partial index for founder's "who is waiting on me?" list
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_debates_awaiting_founder
  ON debates(company_id)
  WHERE status = 'paused' AND awaiting_input_from = 'founder';

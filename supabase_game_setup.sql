-- Guess the Cuber: head-to-head match state.
-- Run this once in the Supabase SQL Editor.
--
-- Why Supabase rather than Flask-SocketIO: the web app runs two Gunicorn
-- workers with no sticky sessions, and the Fly machine suspends when idle
-- (auto_stop_machines = 'suspend'), which drops any open WebSocket. Terminating
-- the socket at Supabase keeps scale-to-zero intact and makes the two-worker
-- problem moot. supabase-js is already loaded on every page and ships the
-- Realtime client, so this adds no frontend dependency.
--
-- The security model: clients SUBSCRIBE to moves but never WRITE them, and the
-- chosen cubers live in a table no client can read. Every write that touches a
-- secret goes through Flask with the service-role key.

-- ---------------------------------------------------------------------------
-- Matches
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS game_matches (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    join_code   TEXT        NOT NULL UNIQUE,
    difficulty  TEXT        NOT NULL DEFAULT 'normal',
    host_user   UUID        NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    guest_user  UUID        REFERENCES auth.users (id) ON DELETE CASCADE,
    -- 'rebuttal': someone has named their opponent's cuber correctly and the
    -- opponent has one guess to level it. `winner` holds the pending winner for
    -- the duration, and is cleared if the rebuttal lands.
    status      TEXT        NOT NULL DEFAULT 'waiting'
                CONSTRAINT game_matches_status_check
                CHECK (status IN ('waiting', 'active', 'rebuttal',
                                  'finished', 'abandoned')),
    turn        UUID,       -- whose turn it is; NULL until both players are in
    winner      UUID,       -- NULL on a draw
    outcome     TEXT        CONSTRAINT game_matches_outcome_check
                CHECK (outcome IN ('win', 'tie', 'resign')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS game_matches_join_code_idx ON game_matches (join_code);
CREATE INDEX IF NOT EXISTS game_matches_players_idx   ON game_matches (host_user, guest_user);

-- ---------------------------------------------------------------------------
-- Secrets — the one table clients must never read
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS game_secrets (
    match_id  UUID NOT NULL REFERENCES game_matches (id) ON DELETE CASCADE,
    user_id   UUID NOT NULL REFERENCES auth.users (id)   ON DELETE CASCADE,
    wca_id    TEXT NOT NULL,
    PRIMARY KEY (match_id, user_id)
);

-- ---------------------------------------------------------------------------
-- Moves — the realtime feed
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS game_moves (
    id          BIGSERIAL PRIMARY KEY,
    match_id    UUID        NOT NULL REFERENCES game_matches (id) ON DELETE CASCADE,
    actor       UUID        NOT NULL REFERENCES auth.users (id)   ON DELETE CASCADE,
    kind        TEXT        NOT NULL
                CONSTRAINT game_moves_kind_check
                CHECK (kind IN ('joined', 'question', 'guess', 'rebuttal',
                                'win', 'tie', 'resign')),
    payload     JSONB       NOT NULL DEFAULT '{}'::JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS game_moves_match_idx ON game_moves (match_id, id);

-- ---------------------------------------------------------------------------
-- Bring an existing database up to date
-- ---------------------------------------------------------------------------
--
-- `CREATE TABLE IF NOT EXISTS` above does nothing to a table that already
-- exists, so a database created before the rebuttal rule keeps its old shape
-- and re-running this file would report success while changing nothing. These
-- statements are what actually applies the change, and they are safe on a
-- fresh database too — the whole file is idempotent, so run it as often as you
-- like.

ALTER TABLE game_matches ADD COLUMN IF NOT EXISTS outcome TEXT;

-- DROP-then-ADD because ADD CONSTRAINT has no IF NOT EXISTS. Dropping a
-- constraint validates nothing and rewrites nothing, so this is cheap.
ALTER TABLE game_matches DROP CONSTRAINT IF EXISTS game_matches_status_check;
ALTER TABLE game_matches ADD  CONSTRAINT game_matches_status_check
    CHECK (status IN ('waiting', 'active', 'rebuttal', 'finished', 'abandoned'));

ALTER TABLE game_matches DROP CONSTRAINT IF EXISTS game_matches_outcome_check;
ALTER TABLE game_matches ADD  CONSTRAINT game_matches_outcome_check
    CHECK (outcome IN ('win', 'tie', 'resign'));

ALTER TABLE game_moves DROP CONSTRAINT IF EXISTS game_moves_kind_check;
ALTER TABLE game_moves ADD  CONSTRAINT game_moves_kind_check
    CHECK (kind IN ('joined', 'question', 'guess', 'rebuttal',
                    'win', 'tie', 'resign'));

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------

ALTER TABLE game_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE game_secrets ENABLE ROW LEVEL SECURITY;
ALTER TABLE game_moves   ENABLE ROW LEVEL SECURITY;

-- Players can read a match they are in. Nobody writes through the client;
-- Flask holds the service-role key, which bypasses RLS by design.
DROP POLICY IF EXISTS game_matches_read ON game_matches;
CREATE POLICY game_matches_read ON game_matches
    FOR SELECT TO authenticated
    USING (auth.uid() = host_user OR auth.uid() = guest_user);

-- Moves are readable by both players in the match. This is what Realtime
-- delivers, and the reason a question's payload carries no answer about the
-- asker's own cuber.
DROP POLICY IF EXISTS game_moves_read ON game_moves;
CREATE POLICY game_moves_read ON game_moves
    FOR SELECT TO authenticated
    USING (EXISTS (
        SELECT 1 FROM game_matches m
         WHERE m.id = game_moves.match_id
           AND (auth.uid() = m.host_user OR auth.uid() = m.guest_user)
    ));

-- No SELECT policy on game_secrets at all. With RLS enabled and no permissive
-- policy, every client read returns zero rows — including the owner's own.
-- Flask is the only thing that ever reads this table. Deliberate: a player
-- has no reason to fetch even their own secret, and the safest policy for a
-- table like this is none.

-- ---------------------------------------------------------------------------
-- Realtime
-- ---------------------------------------------------------------------------

-- Publish moves so clients get postgres_changes over the WebSocket. Matches
-- are published too, so a player sees the opponent join and the turn flip.
-- game_secrets is deliberately absent.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
         WHERE pubname = 'supabase_realtime' AND tablename = 'game_moves'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE game_moves;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
         WHERE pubname = 'supabase_realtime' AND tablename = 'game_matches'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE game_matches;
    END IF;
END $$;

-- Realtime respects RLS for postgres_changes, so a subscriber only receives
-- rows the policies above already let them SELECT.
ALTER TABLE game_moves   REPLICA IDENTITY FULL;
ALTER TABLE game_matches REPLICA IDENTITY FULL;

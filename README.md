# SpeedCubeMuse

An AI-powered tool that lets you query World Cube Association competition data using plain English, and ask WCA rules questions to an AI trained on the official regulations. Available as both a web app and a Discord bot.

**Live site:** [speedcubemuse.fly.dev](https://speedcubemuse.fly.dev)

## Features

- Natural language to SQL translation using Anthropic's Claude AI
- Query WCA statistics using plain English — no SQL required
- **Ask a Delegate** — AI chatbot grounded in WCA Regulations and Guidelines using a RAG pipeline
- **Guess the Cuber** — an Akinator-style guessing game over the WCA competitor pool, in three modes
- Web interface with instant results displayed in formatted tables
- Discord bot with the same query capabilities
- Google and WCA OAuth sign-in via Supabase Auth
- Save and revisit past queries from your profile
- Guest access with limited free queries before sign-in required

## Database

The app queries a database populated from the official [WCA data export](https://www.worldcubeassociation.org/export/results) (August 2, 2026), containing:

| Stat | Count |
|------|-------|
| Competitors | 293,935 |
| Results | 6,764,347 |
| Competitions | 18,280 |
| Events | 17 |

The database is refreshed automatically every Monday by the `Weekly WCA database refresh` GitHub Action, which runs `scripts/update_database.py` to download the latest WCA export and reload all tables. The script records the export date and row counts in a `site_meta` table that the home and About pages read at render time, so a refresh reaches the site with no commit or redeploy.

## Tech Stack

- **Backend:** Python, Flask
- **AI:** Anthropic Claude (natural language to SQL, Ask a Delegate generation)
- **Embeddings & Reranking:** Voyage AI (Ask a Delegate RAG pipeline)
- **WCA Database:** TiDB Serverless (MySQL-compatible)
- **Auth & Saved Queries:** Supabase (PostgreSQL + Auth)
- **Discord:** discord.py
- **Deployment:** Fly.io (one app with separate website and bot machines), Docker, Gunicorn
- **CI/CD:** GitHub Actions

## How It Works

### WCA Data Queries
1. User asks a question in plain English (web or Discord)
2. Claude AI translates the question into a SQL query against the WCA database schema
3. The query executes against TiDB Serverless and results are returned in a formatted table

### Ask a Delegate
1. User asks a WCA rules question in plain English
2. The question is matched against all 697 WCA regulations and guidelines using Voyage AI vector embeddings and pgvector similarity search, then re-ranked
3. Claude generates a grounded response with citations linked to the official WCA Regulations page

### Guess the Cuber

An Akinator-style game over the WCA competitor pool. **The LLM is not the game
engine** — Akinator is an information-gain search over a feature matrix, not a
language model, and this works the same way.

`scripts/build_cuber_profiles.py` precomputes ~37 attributes for **every WCA
competitor** into a `cuber_profiles` table during the weekly refresh — all
~297,000 of them. A `fame` score (records, World Championship podiums, recency,
result volume) ranks the ~2,100 who have held a continental record or better or
sit in a current world top 100, and those form the first three tiers.

| Difficulty | Pool | Rows |
|---|---|---|
| Easy | best-known competitors | 300 |
| Normal | + the next tier of names | 1,000 |
| Hard | every record holder and current world top 100 | 2,097 |
| Everyone | + everyone with 5+ competitions | 51,904 resident |
| *(you guess mine only)* | literally any competitor | 296,987 |

Everyone stops at five competitions for the modes where the *app* guesses,
because that is where the data runs out: 155k competitors have been to exactly
one competition and 50k to two, and their profiles are identical in nearly every
attribute the game can ask about. "You guess mine" has no such limit — it holds
one secret and answers questions about that single row, so it reaches the full
297k via an indexed lookup rather than loading the pool.

Memory is the constraint that shapes this. The Fly machine has 512 MB and runs
two workers, so the Everyone pool loads only when that difficulty is actually
played, and `_shape` folds duplicate strings onto shared objects — `json.loads`
allocates a fresh string per key per row, which measured as the largest single
use of memory in the worker (192 MB → 69 MB once folded).

Two parts of the schema are easy to get wrong, and both are asserted in
`tests/test_game_profiles.py`:

- **Records are a hierarchy — WR > CR > NR.** `has_cr_or_better` is true for
  anyone who has set a world record, even with zero recorded continental
  records. Without this the FMC world record holder answers "no" to "continental
  record or better", which is how the bug was found. "Ever set" and "currently
  holds" are also separate questions: the `currently_*` attributes come from the
  rank tables, where the hierarchy is automatic since rank 1 in the world is
  necessarily rank 1 in your continent and country.
- **Event groups.** `big_cubes` (4x4–7x7), `blind_events` (3BLD, 4BLD, 5BLD,
  MBLD) and `side_events` (Pyraminx, Megaminx, Skewb, Square-1, Clock) are
  defined once in `services/game/attributes.py` and drive both the specialist
  booleans and the group questions, so the two cannot drift apart.
  `top100_events` is set-valued, so "top 100 in 4x4?", "in any blind event?" and
  "in 3x3?" are all one attribute rather than a boolean per event.

Question *selection* is then pure entropy math over that matrix — expected
information gain, no API call. Answers update a Bayesian belief rather than
filtering candidates outright, so one wrong answer damps a candidate instead of
eliminating it and the search can recover.

Predicates are evaluated on demand rather than indexed: a precomputed
predicate→rows map costs ~140 MB at 52k candidates against ~62 MB for the rows
themselves. Question scoring uses at most 3,000 sampled candidates, groups their
probability by attribute value, and scores each question from the total mass on
its two sides. This is the same noisy yes/no information gain as constructing
both full posterior distributions, with far fewer repeated evaluations and
logarithms. The sample size, belief updates, and stopping thresholds are unchanged.

On a local benchmark of the September 22, 2026 pool (52,051 candidates), opening
question selection fell from 151 ms to 12 ms, and turn 10 from 135 ms to 11 ms,
with the same questions selected for the same random sample. These timings do
not reproduce the Fly timeout or measure production latency. The turn endpoint
logs `akinator stage=load`, `stage=replay`, and `stage=pick` with elapsed and CPU
milliseconds, including on a worker abort, to distinguish slow loading from
scoring and help investigate CPU contention.

Measured by self-play:

| Pool | Accuracy | Median questions |
|---|---|---|
| Easy (300) | 100% | 10 |
| Hard (2,097) | 100% | 13 |
| Hard, with 10% of answers wrong | 90% | — |
| Everyone (51,904) | 60% | 26 |

That last row is why the app offers a **shortlist** instead of a single name
when it isn't confident: on the Everyone pool the search regularly runs out of
separating questions with a cluster of near-identical candidates still standing,
and naming one of them would just be a confident wrong answer.

### Never asking what you already answered

The belief update is deliberately soft, and that has a cost: answering "yes,
75+ competitions" leaves ~44% of the probability mass on candidates with fewer,
so "20+ competitions?" still scores as informative even though its answer is
certain. Measured on the ~52k pool, **42% of questions were logically settled by
an earlier answer** — nearly all of them another value of a categorical the
player had already pinned down ("they're Swiss" followed by "are they German?",
across 166 countries).

`Engine.implied()` derives the settled set from the answer log and excludes it
from selection: weaker numeric thresholds after a yes and stronger ones after a
no, every other value of a decided categorical, event groups implied by a member
(and members ruled out by an empty group), and the declared boolean hierarchies
in `Attribute.implies` walked transitively in both directions. That took implied
questions to 0% and cut the median Everyone game from 46 questions to 26, with
accuracy unchanged.

`pick_question` takes the answer log rather than a set of asked ids so this
cannot be bypassed — and, importantly, so the settled set is never confused with
the question count. `should_guess` counts only questions actually put to the
player; folding the settled ids into the same set ends a game after about six
real questions, since pinning down a country settles a hundred-odd others at
once.

### Profile photos

The result card shows the competitor's WCA profile photo, fetched straight from
the browser — the public API sends `access-control-allow-origin: *`, so no proxy
is needed and a ~350ms call to a third party never delays the guess itself. It
only appears once the answer is revealed, never during play.

`avatar.is_default` marks competitors who never uploaded one, whose URL points
at a generic silhouette; those are skipped, and the empty slot collapses so no
gap is left. That branch is common — roughly 75% of the Easy tier has a real
photo against ~40% of the Everyone tier. No CSP change was needed (`img-src`
already allows https, `connect-src` already lists the WCA API), but tests pin
both, since a tightened policy would break this silently.

### When it stops asking

Three conditions end a round, whichever comes first: the leader clears
`GUESS_THRESHOLD` (90%), one candidate is left, or no remaining question
meaningfully separates the field. `MAX_QUESTIONS` (75) is the backstop for a
secret that genuinely cannot be narrowed.

Above the threshold the app names the competitor outright; below it, it offers a
shortlist. Measured across all three pools every outright guess was correct at
both 85% and 90%, so the bar is headroom rather than a fix — it costs about two
extra questions on the largest pool and none on the curated ones, where the
belief clears any of these thresholds in a single answer.

The third condition is the one doing the work, because information gain
collapses as the search narrows — measured on the ~52k pool, the median question
is worth 0.60 bits at Q1, 0.26 at Q15, 0.02 at Q25, 0.001 at Q40 and 0.0001 by
Q74. Running to a fixed cap would mean asking dozens of questions that provably
cannot change the answer, so `MIN_QUESTION_GAIN` stops the search when questions
stop earning their place. On the curated pools it never applies; games there end
on confidence around question 13.

| Mode | Route | Model calls |
|------|-------|-------------|
| **I'll guess yours** — you think of a competitor, the app asks | `/game/akinator` | **none** |
| **You guess mine** — the app hides one, you ask in plain English | `/game/solo` | one Haiku call per typed question |
| **Head to head** — two players race to name each other's | `/game/pvp` | one per typed question |

Only free-text questions reach a model, and only to map wording onto a known
attribute (`"are they retired?"` → `is_active is false`). That is a small
classification task on Haiku with a reusable system prompt, and results are cached
by normalized question text in two tiers — per-process, then a shared
`game_question_cache` table. A repeat question resolves in about a millisecond
and costs nothing.

Rate limits follow the same split. Question selection, name search and guessing
never call a model, so they are exempt or capped only high enough to stop a
client hammering the CPU (`GAME_LIMIT_FREE`, `GAME_LIMIT_CHEAP`); only the two
"ask" endpoints are held tighter (`GAME_LIMIT_ASK`), plus a guest day-cap
(`MAX_GUEST_GAME_QUESTIONS`) that signed-in users are exempt from. That cap is
the one control stopping an anonymous visitor running up an API bill, so prefer
raising it to removing it. `GAME_RATE_LIMITS=false` exempts the game entirely —
note it *exempts* rather than un-decorating, because an undecorated route would
inherit the app-wide 200/day + 50/hour defaults and end up more restricted than
before.

Cache entries carry a fingerprint of the attribute schema and are ignored when
it changes. This matters most for cached *declines*: a question the schema
couldn't answer yesterday becomes answerable the moment an attribute lands, and
a stale "I can't answer that" would otherwise keep refusing it invisibly.
Nothing needs purging by hand.

State is deliberately externalized: the web app runs two Gunicorn workers with no
sticky sessions and the Fly machine suspends when idle, so nothing survives in
process memory between turns. Mode 2 is fully stateless (the client replays its
answer log, and the secret is only ever in the player's head); Mode 1 seals the
secret into a Fernet-encrypted token the browser carries — a Flask session cookie
is signed but *not* encrypted, so a player could simply decode it; head-to-head
keeps match state in Supabase.

## Web App

The web interface provides:
- A search bar to ask any question about WCA data
- Formatted result tables with the generated SQL visible
- **Ask a Delegate** page for WCA rules and regulation questions
- Google and WCA OAuth sign-in for unlimited queries and saved query history
- A profile page with account info, provider badge (Google/WCA), and saved queries

## Discord Bot

### Commands

| Command | Description |
|---------|-------------|
| `/delegate <question>` | Ask about the WCA Regulations & Guidelines (opens a thread for follow-ups) |
| `/query <question>` | Ask about competitors, records, and results |
| `/help` | Show commands and example questions |
| `/ping` | Check whether the bot is responding |

Choose `/query` or `/delegate` from Discord's command menu, then fill in the
`question` field. Existing `!wca query`, `!wca q`, `!wca ask`, `!wca help`, and
`!wca ping` commands still work.

`/delegate` answers cite the official regulations inline (e.g. `[9b1]`) with links to the source text, and each answer opens a thread where follow-up questions keep the conversation context.

### Add to Your Server

1. Use the [invite link](https://discord.com/oauth2/authorize?client_id=1450571905043267594&permissions=309237730304&scope=bot%20applications.commands) to add the bot
2. Select your server (requires **Manage Server** permissions)
3. Authorize the requested permissions (slash commands, sending messages, embeds, and threads)
4. Choose `/query` for results or `/delegate` for rules, then enter your question

## Example Questions

- What is the world record for 3x3?
- Who are the top 10 fastest 2x2 solvers?
- How many competitions have been held in the United States in 2025?
- Who has the most world record single results?
- Who placed first in 3x3 finals at the 2023 World Championship?

## Project Structure

```
wca_statbot/
├── app.py                  # Flask web application
├── config.py               # Configuration management (shared by web + bot)
├── delegate-bot/           # Discord bot — runs on its own machine in speedcubemuse
│   ├── bot.py              # Bot entrypoint (/query and /delegate slash commands)
│   ├── delegate.py         # Embed building + thread conversation history helpers
│   ├── Dockerfile          # Optional standalone bot image; unused by GitHub Actions
│   ├── fly.toml            # Optional separate-app config; unused by GitHub Actions
│   └── requirements.txt    # Bot-only dependencies
├── blueprints/
│   └── game.py             # Guess the Cuber routes (registered on the Flask app)
├── services/
│   ├── nl_to_sql.py        # Natural language to SQL translation (Claude AI)
│   ├── wca_api.py          # WCA database query execution and formatting
│   ├── rag.py              # Ask a Delegate RAG pipeline (Voyage AI + Claude)
│   ├── auth.py             # Supabase authentication helpers
│   ├── saved_queries.py    # Saved query CRUD operations
│   ├── site_meta.py        # DB-backed export date + stats shown on the site
│   └── game/               # Guess the Cuber
│       ├── attributes.py       # Attribute schema — single source of truth
│       ├── engine.py           # Bayesian filter + information-gain picker
│       ├── profiles.py         # cuber_profiles read path + process cache
│       ├── question_parser.py  # Free text -> predicate (the only model call)
│       ├── tokens.py           # Fernet-sealed secrets for solo games
│       └── pvp.py              # Head-to-head match state (Supabase)
├── templates/
│   ├── index.html          # Main query page
│   ├── about.html          # About page
│   ├── delegate.html       # Ask a Delegate page
│   ├── login.html          # Login page (Google / WCA OAuth)
│   └── profile.html        # Profile page with saved queries
├── static/
│   └── style.css           # Styles
├── scripts/
│   ├── update_database.py       # Download and reload WCA data export into TiDB
│   └── build_cuber_profiles.py  # Build the Guess the Cuber feature matrix
├── tests/
│   ├── test_database.py       # Integration tests for database integrity
│   ├── test_game_profiles.py  # Feature-matrix integrity
│   ├── test_game_engine.py    # Engine unit tests + self-play quality gate
│   └── test_game_api.py       # Token sealing + model-output validation
├── .github/
│   └── workflows/
│       ├── deploy.yml            # CI/CD (deploys website and bot together on push)
│       └── update-database.yml   # Weekly WCA export refresh (Mondays 09:00 UTC)
├── Dockerfile              # Shared image for website and bot
├── fly.toml                # speedcubemuse app: website and bot process groups
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not in git)
```

## Local Development

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/)
- [Voyage AI API key](https://www.voyageai.com/) (for Ask a Delegate)
- TiDB Serverless database (or local MySQL) with WCA data imported
- Supabase project (for auth and saved queries)
- Discord bot token (if running the bot)

### Setup

```bash
git clone https://github.com/timcuber37/wca_statbot.git
cd wca_statbot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file:

```env
# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-5

# Voyage AI (Ask a Delegate RAG)
VOYAGE_API_KEY=your_key_here

# Guess the Cuber (optional — these are the defaults)
GAME_MODEL=claude-haiku-4-5
GAME_RATE_LIMITS=true
GAME_LIMIT_FREE=300 per minute
GAME_LIMIT_CHEAP=120 per minute
GAME_LIMIT_ASK=60 per minute
MAX_GUEST_GAME_QUESTIONS=300

# WCA Database (TiDB Serverless)
DB_HOST=gateway01.us-east-1.prod.aws.tidbcloud.com
DB_PORT=4000
DB_USER=your_tidb_user
DB_PASSWORD=your_tidb_password
DB_NAME=wca
DB_SSL=true

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# WCA OAuth
WCA_CLIENT_ID=your_wca_client_id
WCA_CLIENT_SECRET=your_wca_client_secret
WCA_REDIRECT_URI=http://localhost:5000/auth/wca/callback

# Flask
SECRET_KEY=your_secret_key

# Discord (optional, for bot only)
DISCORD_TOKEN=your_discord_token
DISCORD_GUILD_ID=your_guild_id

# App Settings
MAX_QUERY_RESULTS=50
# Optional prefix for legacy text commands; slash commands always use /
COMMAND_PREFIX=!wca
```

### Run the web app

```bash
python app.py
```

### Review API usage and cost

API operations record token usage and estimated USD costs in application logs
and the shared WCA database. After deployment, compare the last month's usage:

```bash
python scripts/usage_report.py --days 30
python scripts/usage_report.py --days 7 --group-by operation,model,source
```

The report separates SQL generation, repair, summaries, Delegate pipeline
steps, game parsing/cache hits, and regulation embedding jobs. See
[API usage and optimization](docs/api-usage.md) for exports, configuration,
billing limitations, and the initial cost audit.

### Run the Discord bot

```bash
pip install -r delegate-bot/requirements.txt
python delegate-bot/bot.py
```

The bot syncs its slash commands with Discord when it starts. Deploy or restart
the bot after adding commands. Set `DISCORD_GUILD_ID` in `.env` while developing
to sync commands to a specific test server; leave it unset for global commands.

### Update the database

Runs weekly on its own (see [Deployment](#deployment)); these are for running it by hand:

```bash
python scripts/update_database.py                 # skip if already up to date
python scripts/update_database.py --force         # reload regardless
python scripts/update_database.py --patch-repo    # also rewrite the stats table above
python scripts/update_database.py --no-fast-load  # strict row-by-row load
```

Tables are bulk-loaded with `LOAD DATA LOCAL INFILE`, ~8x faster than batched
INSERTs (200k rows: 48.3s → 5.7s). Each statement is capped at both 64 MB and
500k rows (`LOAD_CHUNK_BYTES` / `LOAD_CHUNK_ROWS`) — TiDB runs one as a single
transaction, and exceeding its memory limit gets the query cancelled. Both caps
are needed: wide tables hit the byte limit first, narrow ones like
`result_attempts` hit the row limit. A failed chunk re-stages 4x smaller and
retries once before falling back. Because MySQL's LOCAL protocol can't abort
mid-stream, the server coerces a value its column type rejects (a non-numeric
int becomes `0`) rather than raising; rows with the wrong column count are still
dropped during staging. `--no-fast-load` restores the strict row-by-row path,
which is also the automatic fallback if `LOAD DATA` errors.

The freshness check reads the last-loaded export date from the `site_meta` table,
falling back to the local `scripts/.last_export_date` file.

### Build the Guess the Cuber matrix

Runs automatically at the end of `update_database.py`; these are for running it
by hand:

```bash
python scripts/build_cuber_profiles.py             # build and write
python scripts/build_cuber_profiles.py --dry-run   # report only, no writes
python scripts/update_database.py --skip-profiles  # reload WCA data, skip the rebuild
```

`--dry-run` prints the pool size, the tier split, and per-attribute balance. Read
the balance column: a boolean true for under 5% or over 95% of the pool barely
ever splits candidates, and a numeric threshold nothing meets is dead weight.
Retune `thresholds` in `services/game/attributes.py` and re-run — the build takes
about 90 seconds; it profiles every competitor, not just the ranked ones. `tests/test_game_profiles.py` asserts both conditions so a
retuned threshold can't silently rot later.

Head-to-head additionally needs `supabase_game_setup.sql` run once in the
Supabase SQL Editor; the other two modes work without it.

### Run tests

```bash
python -m pytest tests/ -v                    # everything
python -m pytest tests/test_database.py -v    # WCA data integrity
python -m pytest tests/test_game_engine.py -v -s   # includes the self-play gate
python -m unittest discover -s tests -p test_bot_commands.py -v  # offline Discord command checks
```

The self-play tests play the engine against every candidate with a perfect oracle
and assert a median of ≤ 15 questions, then replay with 10% of answers flipped to
confirm the Bayesian update still recovers. That is the real quality gate for the
game: whether ~33 attributes actually separate 2,100 people is not something you
can tell by reading the list. All DB-backed suites skip automatically when the
database is unreachable.

## Deployment

The website and Discord bot run in the existing **`speedcubemuse` Fly.io app**.
The root `fly.toml` defines two [process groups](https://fly.io/docs/launch/processes/),
each with its own machine:

| Process group | Runs | Memory | When idle |
| --- | --- | --- | --- |
| `app` | Flask website | 512 MB | Suspends; web traffic wakes it |
| `bot` | Discord bot | 256 MB | Keeps running |

Only `app` has an HTTP service. The bot has no service managed by Fly Proxy, so
website inactivity does not stop it. Both machines use the root Docker image
and share the app's runtime secrets.

Pushes to `main` deploy both processes in one job using the existing
`FLY_API_TOKEN` GitHub repository secret. The workflow can also be started from
**Actions → Deploy to Fly.io → Run workflow**. A second Fly app or deploy token
is not needed.

### Move to the shared app setup

1. Check the secrets on `speedcubemuse` with `fly secrets list --app speedcubemuse`.
   The bot needs `DISCORD_TOKEN` plus the shared AI and database settings shown
   below. Add any missing values before deploying. `DISCORD_TOKEN` is the
   Discord bot token, separate from the Fly deployment token.

2. Commit and push the changes to `main`, or deploy from the repository root:

   ```bash
   fly deploy --config fly.toml --remote-only --ha=false
   ```

   This keeps the website in its existing `app` process group and adds `bot`.
   `--ha=false` prevents Fly from creating extra machines for redundancy; it
   does not remove replicas that already exist.

3. After deployment, set the counts to one website machine and one bot machine:

   ```bash
   fly scale count app=1 bot=1 --app speedcubemuse
   fly status --app speedcubemuse
   fly logs --app speedcubemuse
   ```

   The scale command removes extra replicas if there are any. Keep exactly one
   bot machine to avoid duplicate Discord responses. Later deployments preserve
   the machine counts.

The previous bot deployment failed with `Error: app not found` because it
targeted `speedcubemuse-bot`, which had not been created. The workflow now uses
the existing app for both processes. The files in `delegate-bot/Dockerfile` and
`delegate-bot/fly.toml` remain available for an optional separate-app setup;
GitHub Actions does not use them.

### Manual deployments and runtime secrets

```bash
# Manual deploy — website and bot together, from the repository root
fly deploy --config fly.toml --remote-only --ha=false

# Shared app secrets (replace placeholders with actual values)
fly secrets set -a speedcubemuse \
  DISCORD_TOKEN=... \
  ANTHROPIC_API_KEY=... \
  VOYAGE_API_KEY=... \
  DB_HOST=... DB_PORT=... DB_USER=... DB_PASSWORD=... DB_NAME=... DB_SSL=true \
  SUPABASE_URL=... SUPABASE_ANON_KEY=... SUPABASE_SERVICE_ROLE_KEY=... \
  WCA_CLIENT_ID=... WCA_CLIENT_SECRET=... WCA_REDIRECT_URI=... \
  SECRET_KEY=...
```

Leave `DISCORD_GUILD_ID` unset in production so slash commands sync globally.
Existing secrets stay in place; only supply values you need to add or change.

### Weekly database refresh

`.github/workflows/update-database.yml` reloads the WCA export every Monday at
09:00 UTC (also runnable on demand from the Actions tab, with an optional
`force` input). It writes only to TiDB — no commit, no redeploy — because the
site reads its stats from the `site_meta` table at render time.

Required repository secrets: `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
(plus optional `DB_PORT`, defaulting to `4000`, and `DISCORD_WEBHOOK_URL` to get
a ping when a run fails).

```bash
gh secret set DB_HOST      # etc.
```

Two notes on GitHub's scheduler: cron runs can be delayed when the platform is
busy, and schedules are disabled automatically after 60 days of repository
inactivity.

## Security

- SQL validation rejects non-SELECT queries and blocks dangerous keywords
- Rate limiting on all API endpoints (Flask-Limiter), with exemptions for authenticated users
- Content Security Policy (CSP) header restricting script, style, and connection sources
- HTML escaping on all user-facing output
- Security headers (X-Content-Type-Options, X-Frame-Options: DENY, Referrer-Policy, Permissions-Policy)
- Input validation, history size limits, and length caps on RAG requests
- Row Level Security on Supabase saved queries table

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built for the speedcubing community.

# SpeedCubeMuse

An AI-powered tool that lets you query World Cube Association competition data using plain English, and ask WCA rules questions to an AI trained on the official regulations. Available as both a web app and a Discord bot.

**Live site:** [speedcubemuse.fly.dev](https://speedcubemuse.fly.dev)

## Features

- Natural language to SQL translation using Anthropic's Claude AI
- Query WCA statistics using plain English — no SQL required
- **Ask a Delegate** — AI chatbot grounded in WCA Regulations and Guidelines using a RAG pipeline
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
- **Deployment:** Fly.io (two apps: web + always-on bot), Docker, Gunicorn
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
| `!wca query <question>` | Ask a question about WCA data |
| `!wca q <question>` | Short alias for query |
| `!wca ask <question>` | Another alias for query |
| `!wca help` | Show available commands |
| `!wca ping` | Check bot latency |

`/delegate` answers cite the official regulations inline (e.g. `[9b1]`) with links to the source text, and each answer opens a thread where follow-up questions keep the conversation context.

### Add to Your Server

1. Use the [invite link](https://discord.com/oauth2/authorize?client_id=1450571905043267594&permissions=309237730304&scope=bot%20applications.commands) to add the bot
2. Select your server (requires **Manage Server** permissions)
3. Authorize the requested permissions (slash commands, sending messages, embeds, and threads)
4. Type `/delegate` or `!wca query` followed by your question in any text channel

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
├── delegate-bot/           # Discord bot — deployed as its own always-on Fly app
│   ├── bot.py              # Bot entrypoint (!wca commands + /delegate slash command)
│   ├── delegate.py         # Embed building + thread conversation history helpers
│   ├── Dockerfile          # Bot image (built with the repo root as context)
│   ├── fly.toml            # speedcubemuse-bot app config (no HTTP service; never suspends)
│   └── requirements.txt    # Bot-only dependencies
├── services/
│   ├── nl_to_sql.py        # Natural language to SQL translation (Claude AI)
│   ├── wca_api.py          # WCA database query execution and formatting
│   ├── rag.py              # Ask a Delegate RAG pipeline (Voyage AI + Claude)
│   ├── auth.py             # Supabase authentication helpers
│   ├── saved_queries.py    # Saved query CRUD operations
│   └── site_meta.py        # DB-backed export date + stats shown on the site
├── templates/
│   ├── index.html          # Main query page
│   ├── about.html          # About page
│   ├── delegate.html       # Ask a Delegate page
│   ├── login.html          # Login page (Google / WCA OAuth)
│   └── profile.html        # Profile page with saved queries
├── static/
│   └── style.css           # Styles
├── scripts/
│   └── update_database.py  # Download and reload WCA data export into TiDB
├── tests/
│   └── test_database.py    # Integration tests for database integrity
├── .github/
│   └── workflows/
│       ├── deploy.yml            # CI/CD (auto-deploys both Fly apps on push)
│       └── update-database.yml   # Weekly WCA export refresh (Mondays 09:00 UTC)
├── Dockerfile              # Web app container
├── fly.toml                # speedcubemuse (web) Fly.io configuration
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
COMMAND_PREFIX=!wca
```

### Run the web app

```bash
python app.py
```

### Run the Discord bot

```bash
pip install -r delegate-bot/requirements.txt
python delegate-bot/bot.py
```

Set `DISCORD_GUILD_ID` in `.env` while developing — slash commands sync to that
guild instantly instead of waiting for global propagation (~1 hour).

### Update the database

Runs weekly on its own (see [Deployment](#deployment)); these are for running it by hand:

```bash
python scripts/update_database.py               # skip if already up to date
python scripts/update_database.py --force       # reload regardless
python scripts/update_database.py --patch-repo  # also rewrite the stats table above
```

The freshness check reads the last-loaded export date from the `site_meta` table,
falling back to the local `scripts/.last_export_date` file.

### Run database integrity tests

```bash
python -m pytest tests/test_database.py -v
```

## Deployment

Deployed on [Fly.io](https://fly.io) as **two apps**. Pushes to `main` automatically deploy both via GitHub Actions.

- **`speedcubemuse`** — the Flask web app. Scale-to-zero (`auto_stop_machines = 'suspend'`); web traffic wakes it.
- **`speedcubemuse-bot`** — the Discord bot. No HTTP service, so the machine never suspends and the bot stays online 24/7. Must run exactly **one** machine (two would open duplicate Discord gateway sessions and answer everything twice).

```bash
# Manual deploy — web app
fly deploy

# Manual deploy — bot (run from the repo root; the root is the build context)
fly deploy . --config delegate-bot/fly.toml --remote-only --ha=false

# Web app secrets
fly secrets set -a speedcubemuse \
  ANTHROPIC_API_KEY=... \
  VOYAGE_API_KEY=... \
  DB_HOST=... DB_PORT=... DB_USER=... DB_PASSWORD=... DB_NAME=... DB_SSL=true \
  SUPABASE_URL=... SUPABASE_ANON_KEY=... SUPABASE_SERVICE_ROLE_KEY=... \
  WCA_CLIENT_ID=... WCA_CLIENT_SECRET=... WCA_REDIRECT_URI=... \
  SECRET_KEY=...

# Bot secrets (no DISCORD_GUILD_ID in prod — commands sync globally)
fly secrets set -a speedcubemuse-bot \
  DISCORD_TOKEN=... \
  ANTHROPIC_API_KEY=... \
  VOYAGE_API_KEY=... \
  DB_HOST=... DB_PORT=... DB_USER=... DB_PASSWORD=... DB_NAME=... DB_SSL=true \
  SUPABASE_URL=... SUPABASE_ANON_KEY=...
```

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

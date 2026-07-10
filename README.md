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

The app queries a database populated from the official [WCA data export](https://www.worldcubeassociation.org/export/results) (July 6, 2026), containing:

| Stat | Count |
|------|-------|
| Competitors | 292,488 |
| Results | 6,693,874 |
| Competitions | 18,077 |
| Events | 17 |

The database is updated using `scripts/update_database.py`, which downloads the latest WCA export, reloads all tables, and automatically patches the stat numbers and export date in the about page.

## Tech Stack

- **Backend:** Python, Flask
- **AI:** Anthropic Claude (natural language to SQL, Ask a Delegate generation)
- **Embeddings & Reranking:** Voyage AI (Ask a Delegate RAG pipeline)
- **WCA Database:** TiDB Serverless (MySQL-compatible)
- **Auth & Saved Queries:** Supabase (PostgreSQL + Auth)
- **Discord:** discord.py
- **Deployment:** Fly.io, Docker, Gunicorn, supervisord
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
| `!wca query <question>` | Ask a question about WCA data |
| `!wca q <question>` | Short alias for query |
| `!wca ask <question>` | Another alias for query |
| `!wca help` | Show available commands |
| `!wca ping` | Check bot latency |

### Add to Your Server

1. Use the [invite link](https://discord.com/oauth2/authorize?client_id=1450571905043267594&permissions=2048&scope=bot) to add the bot
2. Select your server (requires **Manage Server** permissions)
3. Authorize the requested permissions
4. Type `!wca query` followed by your question in any text channel

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
├── bot.py                  # Discord bot
├── config.py               # Configuration management
├── supervisord.conf        # Multi-process supervisor (web + bot)
├── services/
│   ├── nl_to_sql.py        # Natural language to SQL translation (Claude AI)
│   ├── wca_api.py          # WCA database query execution and formatting
│   ├── delegate.py         # Ask a Delegate RAG pipeline (Voyage AI + Claude)
│   ├── auth.py             # Supabase authentication helpers
│   └── saved_queries.py    # Saved query CRUD operations
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
│       └── deploy.yml      # GitHub Actions CI/CD (auto-deploy to Fly.io)
├── Dockerfile              # Docker container for deployment
├── fly.toml                # Fly.io configuration
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
python bot.py
```

### Update the database

```bash
python scripts/update_database.py           # skip if already up to date
python scripts/update_database.py --force   # reload regardless
```

### Run database integrity tests

```bash
python -m pytest tests/test_database.py -v
```

## Deployment

The app is deployed on [Fly.io](https://fly.io) using Docker. The container runs both the web app and Discord bot via supervisord. Pushes to `main` automatically deploy via GitHub Actions.

```bash
# Manual deploy
fly deploy

# Set secrets
fly secrets set \
  ANTHROPIC_API_KEY=... \
  VOYAGE_API_KEY=... \
  DB_HOST=... DB_PORT=... DB_USER=... DB_PASSWORD=... DB_NAME=... DB_SSL=true \
  SUPABASE_URL=... SUPABASE_ANON_KEY=... SUPABASE_SERVICE_ROLE_KEY=... \
  WCA_CLIENT_ID=... WCA_CLIENT_SECRET=... WCA_REDIRECT_URI=... \
  SECRET_KEY=... \
  DISCORD_TOKEN=...
```

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

"""Configuration management for SpeedCubeMuse."""
import os
from dotenv import load_dotenv

load_dotenv()

# Discord Bot Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")  # Optional: for testing in specific server

# Anthropic Configuration (for NL-to-SQL translation)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")  # default model for NL-to-SQL

# User-selectable models for the main query feature.
# Keys are the values the frontend sends; values are the Anthropic model IDs.
# All three must support structured outputs (used by NLToSQLService).
QUERY_MODELS = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}
DEFAULT_QUERY_MODEL = "sonnet"

# WCA API Configuration
WCA_API_BASE_URL = os.getenv("WCA_API_BASE_URL", "https://www.worldcubeassociation.org/api/v0")
# Alternative: If using the unofficial REST API
WCA_REST_API_URL = os.getenv("WCA_REST_API_URL", "https://raw.githubusercontent.com/robiningelbrecht/wca-rest-api/master/api")

# Database Configuration (local MySQL or TiDB Serverless)
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "mysql+aiomysql://user:pass@localhost/wca"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "wca")
DB_SSL = os.getenv("DB_SSL", "false").lower() == "true"  # Required for TiDB Serverless

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# WCA OAuth Configuration
WCA_CLIENT_ID = os.getenv("WCA_CLIENT_ID", "")
WCA_CLIENT_SECRET = os.getenv("WCA_CLIENT_SECRET", "")
WCA_REDIRECT_URI = os.getenv("WCA_REDIRECT_URI", "http://localhost:5000/auth/wca/callback")

# Flask session secret (required for OAuth CSRF state)
SECRET_KEY = os.getenv("SECRET_KEY", "")

# Bot Settings
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!wca")
MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", "50"))

# Ask a Delegate (RAG) Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3-large")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2-lite")
# Pin to a specific commit SHA in production for reproducibility.
REGULATIONS_REF = os.getenv("REGULATIONS_REF", "official")
MAX_GUEST_DELEGATE_QUESTIONS = int(os.getenv("MAX_GUEST_DELEGATE_QUESTIONS", "3"))
DELEGATE_RETRIEVAL_K = int(os.getenv("DELEGATE_RETRIEVAL_K", "6"))
DELEGATE_RERANK_INITIAL_K = int(os.getenv("DELEGATE_RERANK_INITIAL_K", "20"))
DELEGATE_MAX_HISTORY_TURNS = int(os.getenv("DELEGATE_MAX_HISTORY_TURNS", "8"))


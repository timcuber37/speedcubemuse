# Build context is the repo root (deploy with:
#   fly deploy . --config delegate-bot/fly.toml --remote-only --ha=false
# ) so services/ and config.py are shared with the web app without duplication.
FROM python:3.12-slim

WORKDIR /app

COPY delegate-bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY services/ ./services/
COPY delegate-bot/*.py ./

CMD ["python", "-u", "bot.py"]

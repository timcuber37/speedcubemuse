"""Flask web application for SpeedCubeMuse."""
import asyncio
import os
import sys
import json
import logging
import secrets
import urllib.parse
import requests as http_requests
from flask import Flask, render_template, request, jsonify, redirect, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

# aiomysql SSL requires SelectorEventLoop on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config import (MAX_QUERY_RESULTS, SUPABASE_URL, SUPABASE_ANON_KEY,
                    WCA_CLIENT_ID, WCA_CLIENT_SECRET, WCA_REDIRECT_URI, SECRET_KEY,
                    QUERY_MODELS, DEFAULT_QUERY_MODEL)
from services.nl_to_sql import NLToSQLService
from services.wca_api import WCAService
from services.auth import get_user_from_token, find_or_create_wca_user, generate_wca_login_link
from services.rag import DelegateRAGService
from services import saved_queries

_WCA_AUTH_URL = "https://www.worldcubeassociation.org/oauth/authorize"
_WCA_TOKEN_URL = "https://www.worldcubeassociation.org/oauth/token"
_WCA_ME_URL = "https://www.worldcubeassociation.org/api/v0/me"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY or secrets.token_hex(32)

MAX_QUESTION_LENGTH = 2000

# Models that guests (signed-out users) may use. Opus and Sonnet require sign-in.
GUEST_QUERY_MODELS = {'haiku'}

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

nl_to_sql_service = NLToSQLService()
wca_service = WCAService()
delegate_service = DelegateRAGService()

MAX_DELEGATE_HISTORY_MESSAGES = 32
MAX_DELEGATE_HISTORY_CHARS = 12000  # ~3 K tokens; prevents history token-stuffing

# Persistent event loop for async services (avoids "Event loop is closed" errors)
_loop = asyncio.new_event_loop()


def run_async(coro):
    """Run an async coroutine on the persistent event loop."""
    return _loop.run_until_complete(coro)


def get_current_user():
    """Extract user from Authorization header."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        return get_user_from_token(token), token
    return None, None


@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://*.supabase.co https://www.worldcubeassociation.org; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    return response


# --- Pages ---

@app.route('/')
@limiter.exempt
def index():
    return render_template('index.html',
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY)


@app.route('/about')
@limiter.exempt
def about():
    return render_template('about.html',
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY)


@app.route('/login')
@limiter.exempt
def login():
    return render_template('login.html',
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY)


@app.route('/profile')
@limiter.exempt
def profile():
    return render_template('profile.html',
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY)


@app.route('/delegate')
@limiter.exempt
def delegate():
    from config import MAX_GUEST_DELEGATE_QUESTIONS
    return render_template('delegate.html',
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY,
                           max_guest_questions=MAX_GUEST_DELEGATE_QUESTIONS)


# --- API ---

@app.route('/query', methods=['POST'])
@limiter.limit("10 per minute")
def query():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Invalid request.'}), 400

    question = data.get('question', '')
    if not isinstance(question, str):
        return jsonify({'error': 'Invalid question format.'}), 400

    question = question.strip()

    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({'error': f'Question too long (max {MAX_QUESTION_LENGTH} characters).'}), 400

    model = data.get('model')
    if model is not None and not isinstance(model, str):
        return jsonify({'error': 'Invalid model format.'}), 400

    # Opus and Sonnet are restricted to signed-in users; guests get Haiku only.
    effective_model = model if model in QUERY_MODELS else DEFAULT_QUERY_MODEL
    if effective_model not in GUEST_QUERY_MODELS:
        user, _ = get_current_user()
        if user is None:
            return jsonify({'error': 'Sign in to use Opus or Sonnet. Guests can use Haiku.'}), 403

    try:
        sql_query = run_async(nl_to_sql_service.translate_to_sql(question, model=model))

        if not sql_query:
            return jsonify({'error': 'Could not translate your question to a safe SQL query.'}), 400

        logger.info(f"Generated SQL: {sql_query}")

        results = run_async(wca_service.execute_query(sql_query))
        html_table = wca_service.format_results_html(results, max_results=MAX_QUERY_RESULTS)
        summary = run_async(nl_to_sql_service.summarize_results(question, results, model=model))

        return jsonify({
            'html': html_table,
            'sql': sql_query,
            'count': len(results),
            'summary': summary,
        })

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({'error': 'Unable to process your query. Please try again.'}), 500


def _is_delegate_authenticated() -> bool:
    """Return True if the current request carries a valid Supabase token."""
    user, _ = get_current_user()
    return user is not None


@app.route('/api/delegate/ask', methods=['POST'])
@limiter.limit("5 per minute")
@limiter.limit("10 per day", exempt_when=_is_delegate_authenticated)
def delegate_ask():
    if not delegate_service.is_ready():
        return jsonify({'error': 'Ask a Delegate is not configured on this server.'}), 503

    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'Invalid request.'}), 400

    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({'error': f'Question too long (max {MAX_QUESTION_LENGTH} characters).'}), 400

    history = data.get('history') or []
    if not isinstance(history, list):
        return jsonify({'error': 'Invalid history format.'}), 400

    # Sanitize history: only accept user/assistant roles, cap per-message length,
    # and reject any history where a client claims an assistant role with suspicious content.
    sanitized = []
    for msg in history[-MAX_DELEGATE_HISTORY_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role')
        content = msg.get('content')
        if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
            sanitized.append({'role': role, 'content': content.strip()[:MAX_QUESTION_LENGTH]})

    # Reject oversized history payloads to prevent token-stuffing.
    total_history_chars = sum(len(m['content']) for m in sanitized)
    if total_history_chars > MAX_DELEGATE_HISTORY_CHARS:
        return jsonify({'error': 'Conversation history too long. Please clear and start a new conversation.'}), 400

    try:
        result = delegate_service.answer(sanitized, question)
        return jsonify({
            'answer': result['answer'],
            'sources': result['sources'],
        })
    except Exception as e:
        logger.error(f"Delegate ask error: {e}", exc_info=True)
        return jsonify({'error': 'Unable to answer right now. Please try again.'}), 500


@app.route('/api/save-query', methods=['POST'])
@limiter.limit("20 per minute")
def save_query():
    user, token = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    data = request.get_json()
    question = data.get('question', '')
    sql_query = data.get('sql', '')
    results = data.get('results', [])

    if not question or not sql_query:
        return jsonify({'error': 'Missing question or SQL'}), 400

    saved = saved_queries.save_query(token, user.id, question, sql_query, results)
    if saved:
        return jsonify({'success': True, 'id': saved['id']})
    return jsonify({'error': 'Failed to save query'}), 500


@app.route('/api/saved-queries', methods=['GET'])
@limiter.limit("30 per minute")
def get_saved():
    user, token = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    queries = saved_queries.get_saved_queries(token, user.id)
    return jsonify({'queries': queries})


@app.route('/api/saved-query/<query_id>', methods=['DELETE'])
@limiter.limit("20 per minute")
def delete_query(query_id):
    user, token = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401

    success = saved_queries.delete_saved_query(token, user.id, query_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'error': 'Failed to delete query'}), 500


# --- WCA OAuth ---

@app.route('/auth/wca')
@limiter.exempt
def wca_login():
    """Redirect the browser to WCA's OAuth authorization page."""
    if not WCA_CLIENT_ID:
        return redirect('/login?error=wca_not_configured')

    state = secrets.token_urlsafe(32)
    session['wca_oauth_state'] = state

    params = {
        'client_id': WCA_CLIENT_ID,
        'redirect_uri': WCA_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'public email',
        'state': state,
    }
    return redirect(_WCA_AUTH_URL + '?' + urllib.parse.urlencode(params))


@app.route('/auth/wca/callback')
@limiter.exempt
def wca_callback():
    """Handle the OAuth callback from WCA, create a Supabase session, and redirect home."""
    error = request.args.get('error')
    if error:
        logger.warning(f"WCA OAuth error: {error}")
        return redirect('/login?error=' + urllib.parse.quote(error))

    state = request.args.get('state', '')
    if not state or state != session.pop('wca_oauth_state', None):
        return redirect('/login?error=invalid_state')

    code = request.args.get('code')
    if not code:
        return redirect('/login?error=no_code')

    # Exchange authorization code for WCA access token.
    try:
        token_resp = http_requests.post(_WCA_TOKEN_URL, data={
            'code': code,
            'client_id': WCA_CLIENT_ID,
            'client_secret': WCA_CLIENT_SECRET,
            'redirect_uri': WCA_REDIRECT_URI,
            'grant_type': 'authorization_code',
        }, timeout=10)
        token_resp.raise_for_status()
        wca_access_token = token_resp.json().get('access_token')
    except Exception as e:
        logger.error(f"WCA token exchange failed: {e}")
        return redirect('/login?error=token_exchange_failed')

    if not wca_access_token:
        return redirect('/login?error=no_access_token')

    # Fetch WCA user profile.
    try:
        me_resp = http_requests.get(_WCA_ME_URL, headers={
            'Authorization': f'Bearer {wca_access_token}'
        }, timeout=10)
        me_resp.raise_for_status()
        wca_user = me_resp.json().get('me', {})
    except Exception as e:
        logger.error(f"WCA user info fetch failed: {e}")
        return redirect('/login?error=user_info_failed')

    email = wca_user.get('email')
    if not email:
        return redirect('/login?error=no_email')

    wca_id = wca_user.get('wca_id', '')
    name = wca_user.get('name', '')
    avatar_url = (wca_user.get('avatar') or {}).get('url', '')

    # Find or create a matching Supabase account for this WCA user.
    user = find_or_create_wca_user(email, name, wca_id, avatar_url)
    if not user:
        return redirect('/login?error=user_creation_failed')

    # Generate a Supabase magic link so the browser gets a real session.
    # The magic link redirects to /login where onAuthStateChange picks it up.
    redirect_to = request.url_root.rstrip('/') + '/login'
    login_link = generate_wca_login_link(email, redirect_to)
    if not login_link:
        return redirect('/login?error=login_link_failed')

    return redirect(login_link)


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true', port=5000)

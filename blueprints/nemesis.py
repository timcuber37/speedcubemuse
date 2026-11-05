"""Public, deterministic WCA nemesis and head-to-head comparisons."""
from functools import wraps
import logging
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request
import pymysql

from config import SUPABASE_URL, SUPABASE_ANON_KEY
from extensions import limiter
from services.nemesis import NemesisService
from services.site_meta import get_site_meta

nemesis_bp = Blueprint('nemesis', __name__)
service = NemesisService()
logger = logging.getLogger(__name__)


def _api(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        try:
            return jsonify(view(*args, **kwargs))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except LookupError as exc:
            return jsonify(error=str(exc)), 404
        except pymysql.MySQLError:
            logger.exception('Nemesis database query failed')
            return jsonify(error='Comparisons are unavailable right now. Please try again shortly.'), 503
    return wrapped


@nemesis_bp.get('/nemesis')
@limiter.exempt
def index():
    static_dir = Path(nemesis_bp.root_path).parent / 'static'
    asset_versions = {name: (static_dir / name).stat().st_mtime_ns
                      for name in ('nemesis.js', 'nemesis.css')}
    return render_template('nemesis.html', supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY, db_meta=get_site_meta(),
                           asset_versions=asset_versions)


@nemesis_bp.get('/api/nemesis/search')
@limiter.limit('120 per minute')
@_api
def search():
    return {'competitors': service.search(request.args.get('q', ''))}


@nemesis_bp.get('/api/nemesis/compare')
@limiter.limit('60 per minute')
@_api
def compare():
    return service.compare(request.args.get('left', ''), request.args.get('right', ''))


@nemesis_bp.get('/api/nemesis/<wca_id>')
@limiter.limit('20 per minute')
@_api
def overview(wca_id):
    try:
        page = int(request.args.get('page', '1'))
    except ValueError:
        raise ValueError('Page must be a whole number.') from None
    return service.overview(wca_id, direction=request.args.get('direction', 'nemeses'), page=page)

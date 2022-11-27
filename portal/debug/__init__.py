from flask import Blueprint

from portal.debug.debug import main, page, video_feed, list_pages

debug = Blueprint('debug', __name__, template_folder='templates', url_prefix='/debug')
debug.route('/')(main)
debug.route('/<name>')(page)
debug.route('/<name>/video_feed')(video_feed)
debug.route('/list_pages')(list_pages)

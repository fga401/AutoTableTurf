from flask import Blueprint
from portal.home.home import main, source_on_change, video_feed

home = Blueprint('home', __name__, template_folder='templates')
home.route('/')(main)
home.route('/sourceOnChange', methods=['POST'])(source_on_change)
home.route('/video_feed')(video_feed)
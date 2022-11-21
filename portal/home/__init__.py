from flask import Blueprint
from portal.home.home import main, change_source, video_feed, key_press

home = Blueprint('home', __name__, template_folder='templates')
home.route('/')(main)
home.route('/source', methods=['POST'])(change_source)
home.route('/video_feed')(video_feed)
home.route('/keypress', methods=['POST'])(key_press)

from flask import Blueprint

from portal.home.home import main, change_source, video_feed, key_press, run, stop, connect_controller

home = Blueprint('home', __name__, template_folder='templates', url_prefix='/home')
home.route('/')(main)
home.route('/source', methods=['POST'])(change_source)
home.route('/video_feed')(video_feed)
home.route('/keypress', methods=['POST'])(key_press)
home.route('/run', methods=['POST'])(run)
home.route('/stop', methods=['POST'])(stop)
home.route('/connect_controller', methods=['POST'])(connect_controller)

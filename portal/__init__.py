from flask import Flask, redirect, url_for

from portal.debug import debug
from portal.home import home

app = Flask(__name__)
app.register_blueprint(home)
app.register_blueprint(debug)


@app.route('/')
def hello_world():
    return redirect(url_for('home.main'))

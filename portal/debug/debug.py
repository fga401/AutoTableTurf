from flask import Response, render_template, jsonify

from portal.debug.debugger import web_debugger


def main():
    return Response(render_template(
        'debug.html',
        pages=web_debugger.list()
    ))


def page(name):
    return Response(render_template(
        'page.html',
        name=name,
    ))


def list_pages():
    pages = [page for page in web_debugger.list()]
    return jsonify(pages)


def generate_frames(name: str):
    while True:
        buf = web_debugger.get(name)
        frame = bytes(buf.getbuffer())
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


def video_feed(name):
    return Response(generate_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')

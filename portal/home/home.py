import io

import cv2
import numpy as np
from flask import Response, request, render_template

from capture import VideoCapture
from controller import DummyController, Controller
from logger import logger
from portal.debug.debugger import web_debugger
from portal.home.capture import ThreadSafeCapture
from portal.home.keymap import keymap
from tableturf.ai import SimpleAI
from tableturf.manager import TableTurfManager, Exit


def list_available_source():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


available_sources = list_available_source()
capture = ThreadSafeCapture(VideoCapture(0))

controller = DummyController()


def main():
    global available_sources
    available_sources = list_available_source()
    return Response(render_template(
        'home.html',
        url=request.url,
        sources=available_sources,
        keymap=keymap
    ))


def run():
    logger.debug(f'portal.home.run')
    ai = SimpleAI()
    manager = TableTurfManager(
        capture,
        controller,
        ai,
        Exit(max_battle=1),
        debug=web_debugger,
    )
    manager.run(deck=0)
    return Response()


def change_source():
    global capture
    source = int(request.json['source'])
    capture.update_capture(VideoCapture(source))
    logger.debug(f'portal.home.source_on_change: source={source}')
    return Response()


def generate_frames():
    empty = np.zeros((1080, 1920, 3))
    _, empty_buf = cv2.imencode(".jpeg", empty)
    empty_frame = bytes(io.BytesIO(empty_buf).getbuffer())
    while True:
        try:
            img = capture.capture()
            _, buffer = cv2.imencode(".jpeg", img)
            frame = bytes(io.BytesIO(buffer).getbuffer())
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        except Exception:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_frame + b'\r\n'


def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def key_press():
    event_type = request.json['type']
    if event_type == 'keydown':
        raw = request.json['key']
        key = keymap.get(raw, None)
        if key is not None:
            if key == Controller.Stick.L_STICK:
                if raw == 'a':
                    controller.tilt_stick(key, -100, 0)
                elif raw == 's':
                    controller.tilt_stick(key, 0, -100)
                elif raw == 'd':
                    controller.tilt_stick(key, 100, 0)
                elif raw == 'w':
                    controller.tilt_stick(key, 0, 100)
            else:
                controller.press_buttons([key])
    # TODO: support long press
    return Response()

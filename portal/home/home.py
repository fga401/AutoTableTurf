import io

import cv2
from flask import Response, request, render_template

from capture import VideoCapture
from controller import DummyController, Controller
from logger import logger
from portal.home.keymap import keymap


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
capture = VideoCapture(0)
controller = DummyController()
empty_buf = io.BytesIO().getbuffer()


def main():
    global available_sources
    available_sources = list_available_source()
    return Response(render_template(
        'index.html',
        url=request.url,
        sources=available_sources,
        keymap=keymap
    ))


def change_source():
    global capture
    capture.close()
    source = request.json['source']
    capture = VideoCapture(source)
    logger.debug(f'portal.home.source_on_change: source={source}')
    return Response()


def generate_frames():
    while True:
        try:
            img = capture.capture()
            _, buffer = cv2.imencode(".jpeg", img)
            buf = io.BytesIO(buffer)
            frame = buf.getbuffer()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            # TODO: fix this
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_buf + b'\r\n')


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

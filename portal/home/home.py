import io
import time

import cv2
from flask import Response, request, render_template

from capture import VideoCapture
from logger import logger


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
empty_buf = io.BytesIO().getbuffer()


def generate_frames():
    while True:
        try:
            img = capture.capture()
            _, buffer = cv2.imencode(".jpeg", img)
            buf = io.BytesIO(buffer)
            frame = buf.getbuffer()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + empty_buf + b'\r\n')


def main():
    global available_sources
    available_sources = list_available_source()
    return Response(render_template(
        'index.html',
        url=request.url,
        sources=available_sources,
    ))


def source_on_change():
    global capture
    capture.close()
    source = request.json['source']
    capture = VideoCapture(source)
    logger.debug(f'portal.home.source_on_change: source={source}')
    return Response()


def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

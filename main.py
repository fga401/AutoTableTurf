import os

from capture import FileLoader
from controller import DummyController
from tableturf.ai import SimpleAI
from tableturf.manager import TableTurfManager, Exit

endpoint = "http://192.168.50.101:5000/"

if __name__ == '__main__':
    # capture = VideoCapture(0)
    path = os.path.join(os.path.realpath(__file__), '..', 'temp', 'redraw')
    capture = FileLoader(path=path)
    # controller = NxbtController(endpoint)
    controller = DummyController()
    ai = SimpleAI()
    manager = TableTurfManager(
        capture,
        controller,
        ai,
        Exit(max_battle=1),
        debug=True,
    )
    manager.run(2)

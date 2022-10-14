from controller import NxbtController
from capture import VideoCapture
from tableturf.ai import SimpleAI
from tableturf.manager import TableTureManager

endpoint = "http://192.168.50.101:5000/"

if __name__ == '__main__':
    screen_capture = VideoCapture(0)
    controller = NxbtController(endpoint)
    ai = SimpleAI()
    manager = TableTureManager(screen_capture, controller, ai, TableTureManager.Closer(max_battle=1))
    manager.run(my_deck_pos=1)

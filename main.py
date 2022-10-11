from controller import DummyController
from screen import WindowScreenCapturer
from tableturf.ai import SimpleAI
from tableturf.manager import TableTureManager

if __name__ == '__main__':
    screen_capturer = WindowScreenCapturer()
    controller = DummyController()
    ai = SimpleAI()
    manager = TableTureManager(screen_capturer, controller, ai, TableTureManager.Closer(max_battle=1))
    manager.run(my_deck_pos=1)

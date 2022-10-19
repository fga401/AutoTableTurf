from tableturf.ai import AI
from tableturf.model import Status, Step


class SimpleAI(AI):
    def redraw(self, status: Status) -> bool:
        return True

    def next_step(self, status: Status) -> Step:
        pass

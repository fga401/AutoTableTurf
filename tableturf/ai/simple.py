import random
from typing import List, Optional

from logger import logger
from tableturf.ai import AI
from tableturf.model import Status, Step, Stage, Card


class SimpleAI(AI):
    def redraw(self, hands: List[Card], stage: Optional[Stage] = None, my_deck: Optional[List[Card]] = None, his_deck: Optional[List[Card]] = None) -> bool:
        return True

    def next_step(self, status: Status) -> Step:
        step = random.choice(status.get_possible_steps())
        logger.debug(f'SimpleAI.next_step: return={step}')
        return step

from controller import Controller
from logger import logger
from tableturf.manager.action import util


def move_deck_cursor_marco(target: int, current: int):
    logger.debug(f'action.move_deck_cursor_marco: target={target}, current={current}')
    current_x = current // 8
    current_y = current % 8
    target_x = target // 8
    target_y = target % 8
    buttons = []
    if current_x > target_x:
        buttons += [Controller.Button.DPAD_LEFT] * (current_x - target_x)
    else:
        buttons += [Controller.Button.DPAD_RIGHT] * (target_x - current_x)
    if current_y > target_y:
        buttons += [Controller.Button.DPAD_UP] * (current_y - target_y)
    else:
        buttons += [Controller.Button.DPAD_DOWN] * (target_y - current_y)
    return util.buttons_to_marco(buttons)

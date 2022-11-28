import numpy as np

from controller import Controller
from logger import logger
from tableturf.manager.action import util
from tableturf.model import Pattern, Grid, Stage, Step


def rotate_card_marco(rotate: int) -> str:
    return util.buttons_to_marco([Controller.Button.Y] * rotate)


def __remove_special_squares(pattern: Pattern):
    if pattern is None:
        return None
    ink = pattern.grid.copy()
    ink[ink != Grid.MyInk.value] = Grid.Empty.value
    if np.all(ink == Grid.Empty.value):
        return None
    return Pattern(ink)


def compare_pattern(a: Pattern, b: Pattern) -> bool:
    if a == b:
        return True
    a = __remove_special_squares(a)
    b = __remove_special_squares(b)
    return a == b


def move_card_marco(current: np.ndarray, preview: Pattern, stage: Stage, step: Step) -> str:
    target = step.pos.copy()
    logger.debug(f'action.move_card_marco: target={target}, current={current}')
    expected_pattern = step.card.get_pattern(step.rotate)
    if preview != expected_pattern:
        # compare trivial squares
        actual_ink = __remove_special_squares(preview)
        expected_ink = __remove_special_squares(expected_pattern)
        if expected_ink is None or actual_ink != expected_ink:
            logger.warn(f'action.move_card_marco: unmatch pattern')
            return ''
        current_offset = preview.offset[np.argmax(preview.squares == Grid.MyInk.value)]
        target_offset = expected_pattern.offset[np.argmax(expected_pattern == Grid.MyInk.value)]
        current = current + current_offset
        target = target + target_offset
        logger.debug(f'action.move_card_marco: fix current. offset={current_offset}, current={current}')
        logger.debug(f'action.move_card_marco: fix target. offset={target_offset}, target={target}')
    diff_y, diff_x = target - current
    if diff_x > 0:
        step_x = 1
        buttons_x = [Controller.Button.DPAD_RIGHT] * diff_x
    else:
        step_x = -1
        buttons_x = [Controller.Button.DPAD_LEFT] * -diff_x
    if diff_y > 0:
        step_y = 1
        buttons_y = [Controller.Button.DPAD_DOWN] * diff_y
    else:
        step_y = -1
        buttons_y = [Controller.Button.DPAD_UP] * -diff_y
    stage_grid = stage.grid
    row_first_wall_count = np.sum(stage_grid[current[0], current[1]:target[1] + 1:step_x] == Grid.Wall.value) + np.sum(stage_grid[current[0]:target[0] + 1:step_y, target[1]] == Grid.Wall.value)
    col_first_wall_count = np.sum(stage_grid[current[0]:target[0] + 1:step_y, current[1]] == Grid.Wall.value) + np.sum(stage_grid[target[0], current[1]:target[1] + 1:step_x] == Grid.Wall.value)
    if row_first_wall_count < col_first_wall_count:
        buttons = buttons_x + buttons_y
    else:
        buttons = buttons_y + buttons_x
    return util.buttons_to_marco(buttons)

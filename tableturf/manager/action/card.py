import numpy as np

from controller import Controller
from logger import logger
from tableturf.manager.action import util
from tableturf.model import Pattern, Grid, Stage, Step


def rotate_card_marco(step: Step) -> str:
    return util.buttons_to_marco([Controller.Button.Y] * step.rotate)


def move_card_marco(current: np.ndarray, preview: Pattern, stage: Stage, step: Step) -> str:
    target = step.pos.copy()
    logger.debug(f'action.move_card_marco: target={target}, current={current}')
    expected_pattern = step.card.get_pattern(step.rotate)
    if preview != expected_pattern:
        # compare trivial squares
        actual_ink = preview.grid.copy()
        actual_ink[actual_ink != Grid.MyInk.value] = Grid.Empty.value
        expected_ink = expected_pattern.grid.copy()
        expected_ink[expected_ink != Grid.MyInk.value] = Grid.Empty.value
        if np.all(actual_ink == Grid.Empty.value) or np.all(expected_ink == Grid.Empty.value):
            logger.warn(f'action.move_card_marco: unmatch pattern')
            return ''
        actual_ink_pattern = Pattern(actual_ink)
        expected_ink_pattern = Pattern(expected_ink)
        if actual_ink_pattern != expected_ink_pattern:
            logger.warn(f'action.move_card_marco: unmatch pattern')
            return ''
        offset = preview.offset[1] if preview.squares[0] == Grid.MySpecial.value else preview.offset[0]
        current = current + offset
        target = target + offset
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

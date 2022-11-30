import numpy as np

from tableturf.model import Stage, Step


def move(stage: Stage, step: Step):
    if step.action == Step.Action.Skip:
        return stage
    grid = stage.grid.copy()
    pattern = step.card.get_pattern(step.rotate)
    offset = pattern.offset + step.pos[np.newaxis, ...]
    grid[offset[:, 0], offset[:, 1]] = pattern.squares
    return Stage(grid)

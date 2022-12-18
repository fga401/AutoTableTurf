import unittest

import numpy as np
from tableturf.ai.alpha.next_step.estimator import Estimator

from tableturf.model import Stage, Grid, Card, Step


class TestAIUtil(unittest.TestCase):

    def test_move(self):
        stage = Stage(np.array([
            [Grid.Wall.value, Grid.Empty.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.HisInk.value, Grid.Empty.value],
            [Grid.MySpecial.value, Grid.Empty.value, Grid.MyInk.value]
        ]))
        card = Card(
            np.array([
                [Grid.MyInk, Grid.MyInk],
                [Grid.Empty, Grid.MySpecial],
            ]),
            5
        )
        step = Step(
            Step.Action.SpecialAttack,
            card,
            2,
            np.array([1, 1])
        )
        next_stage = Estimator().get_stage(stage, step)
        expected = np.array([
            [Grid.Wall.value, Grid.Empty.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.MySpecial.value, Grid.Empty.value],
            [Grid.MySpecial.value, Grid.MyInk.value, Grid.MyInk.value]
        ])
        self.assertTrue(np.all(next_stage.grid == expected))

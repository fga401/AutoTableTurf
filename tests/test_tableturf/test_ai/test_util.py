import unittest

import numpy as np

import tableturf.ai.util
from tableturf.model import Stage, Grid, Card, Step


class TestCard(unittest.TestCase):

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
        next_stage = tableturf.ai.util.move(stage, step)
        expected = np.array([
            [Grid.Wall.value, Grid.Empty.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.MySpecial.value, Grid.Empty.value],
            [Grid.MySpecial.value, Grid.MyInk.value, Grid.MyInk.value]
        ])
        self.assertTrue(np.all(next_stage.grid == expected))


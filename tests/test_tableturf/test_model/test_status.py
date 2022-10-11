import unittest

import numpy as np

from tableturf.model import Stage, Grid, Card, Status, Step


class TestStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Stage(np.array([
            [Grid.Wall.value, Grid.Empty.value, Grid.HisSpecial.value],
            [Grid.Gray.value, Grid.MySpecial.value, Grid.MyInk.value],
            [Grid.Empty.value, Grid.Empty.value, Grid.HisInk.value],
        ]))
        self.card_0 = Card(0, np.array([
            [Grid.Empty.value, Grid.Empty.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.MySpecial.value, Grid.MyInk.value],
            [Grid.Empty.value, Grid.Empty.value, Grid.Empty.value],
        ]), 1)
        self.card_1 = Card(1, np.array([
            [Grid.Empty.value, Grid.Empty.value, Grid.MyInk.value],
            [Grid.Empty.value, Grid.MySpecial.value, Grid.MyInk.value],
            [Grid.Empty.value, Grid.Empty.value, Grid.Empty.value],
        ]), 2)
        self.card_2 = Card(2, np.array([
            [Grid.Empty.value, Grid.Empty.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.MySpecial.value, Grid.Empty.value],
            [Grid.Empty.value, Grid.Empty.value, Grid.Empty.value],
        ]), 4)
        self.status = Status(
            stage=self.stage,
            hands=[self.card_0, self.card_1, self.card_2],
            my_sp=2,
            his_sp=0,
            my_deck=[],
            his_deck=[],
        )

    def test_status_steps(self):
        steps_0 = self.status.get_possible_steps(0)
        self.assertIn(Step(self.card_0, rotate=0, pos=np.array([2, 0]), action=Step.Action.Place), steps_0)
        self.assertIn(Step(self.card_0, rotate=2, pos=np.array([2, 1]), action=Step.Action.Place), steps_0)
        self.assertIn(Step(self.card_0, rotate=0, pos=np.array([2, 0]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=0, pos=np.array([2, 1]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=2, pos=np.array([2, 1]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=2, pos=np.array([2, 2]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=1, pos=np.array([2, 2]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=3, pos=np.array([1, 2]), action=Step.Action.SpecialAttack), steps_0)
        self.assertIn(Step(self.card_0, rotate=None, pos=None, action=Step.Action.Skip), steps_0)
        self.assertEqual(len(steps_0), 9)

        steps_1 = self.status.get_possible_steps(1)
        self.assertIn(Step(self.card_1, rotate=0, pos=np.array([2, 1]), action=Step.Action.SpecialAttack), steps_1)
        self.assertIn(Step(self.card_1, rotate=None, pos=None, action=Step.Action.Skip), steps_1)
        self.assertEqual(len(steps_1), 2)

        steps_2 = self.status.get_possible_steps(2)
        for rotate in range(4):
            self.assertIn(Step(self.card_2, rotate=rotate, pos=np.array([0, 1]), action=Step.Action.Place), steps_2)
            self.assertIn(Step(self.card_2, rotate=rotate, pos=np.array([2, 0]), action=Step.Action.Place), steps_2)
            self.assertIn(Step(self.card_2, rotate=rotate, pos=np.array([2, 1]), action=Step.Action.Place), steps_2)
        self.assertIn(Step(self.card_2, rotate=None, pos=None, action=Step.Action.Skip), steps_2)
        self.assertEqual(len(steps_2), 13)

        self.assertEqual(len(self.status.get_possible_steps()), 24)

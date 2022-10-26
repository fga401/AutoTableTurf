import unittest

import numpy as np

from tableturf.model import Stage, Grid, Card, Status, Step


class TestStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.HisSpecial],
            [Grid.Gray, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.HisInk],
        ]))
        self.card_0 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]), 1)
        self.card_1 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.MyInk],
            [Grid.Empty, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]), 2)
        self.card_2 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.MySpecial, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty],
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
        steps_0 = self.status.get_possible_steps(self.card_0)
        self.assertIn(Step(Step.Action.Place, self.card_0, rotate=0, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.Place, self.card_0, rotate=2, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=0, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=0, pos=np.array([2, 1])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=2, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=2, pos=np.array([2, 1])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=1, pos=np.array([1, 2])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_0, rotate=3, pos=np.array([1, 2])), steps_0)
        self.assertIn(Step(Step.Action.Skip, self.card_0, rotate=None, pos=None), steps_0)
        self.assertEqual(len(steps_0), 9)

        steps_1 = self.status.get_possible_steps(self.card_1)
        self.assertIn(Step(Step.Action.SpecialAttack, self.card_1, rotate=0, pos=np.array([1, 2])), steps_1)
        self.assertIn(Step(Step.Action.Skip, self.card_1, rotate=None, pos=None), steps_1)
        self.assertEqual(len(steps_1), 2)

        steps_2 = self.status.get_possible_steps(self.card_2)
        self.assertIn(Step(Step.Action.Place, self.card_2, rotate=0, pos=np.array([0, 1])), steps_2)
        self.assertIn(Step(Step.Action.Place, self.card_2, rotate=0, pos=np.array([2, 0])), steps_2)
        self.assertIn(Step(Step.Action.Place, self.card_2, rotate=0, pos=np.array([2, 1])), steps_2)
        self.assertIn(Step(Step.Action.Skip, self.card_2, rotate=None, pos=None), steps_2)
        self.assertEqual(len(steps_2), 4)

        self.assertEqual(len(self.status.get_possible_steps()), 15)

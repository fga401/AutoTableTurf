import unittest

import numpy as np

from tableturf.model import Stage, Grid, Card, Status, Step


class TestStatus(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_status_steps_v1(self):
        stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.HisSpecial],
            [Grid.Neutral, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.HisInk],
        ]))
        card_0 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]), 1)
        card_1 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.MyInk],
            [Grid.Empty, Grid.MySpecial, Grid.MyInk],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]), 2)
        card_2 = Card(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.MySpecial, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]), 4)
        status = Status(
            stage=stage,
            hands=[card_0, card_1, card_2],
            round=12,
            my_sp=2,
            his_sp=0,
            my_deck=[],
            his_deck=[],
        )
        steps_0 = status.get_possible_steps(card_0)
        self.assertIn(Step(Step.Action.Place, card_0, rotate=0, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.Place, card_0, rotate=2, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=0, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=0, pos=np.array([2, 1])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=2, pos=np.array([2, 0])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=2, pos=np.array([2, 1])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=1, pos=np.array([1, 2])), steps_0)
        self.assertIn(Step(Step.Action.SpecialAttack, card_0, rotate=3, pos=np.array([1, 2])), steps_0)
        self.assertIn(Step(Step.Action.Skip, card_0, rotate=None, pos=None), steps_0)
        self.assertEqual(len(steps_0), 9)

        steps_1 = status.get_possible_steps(card_1)
        self.assertIn(Step(Step.Action.SpecialAttack, card_1, rotate=0, pos=np.array([1, 2])), steps_1)
        self.assertIn(Step(Step.Action.Skip, card_1, rotate=None, pos=None), steps_1)
        self.assertEqual(len(steps_1), 2)

        steps_2 = status.get_possible_steps(card_2)
        self.assertIn(Step(Step.Action.Place, card_2, rotate=0, pos=np.array([0, 1])), steps_2)
        self.assertIn(Step(Step.Action.Place, card_2, rotate=0, pos=np.array([2, 0])), steps_2)
        self.assertIn(Step(Step.Action.Place, card_2, rotate=0, pos=np.array([2, 1])), steps_2)
        self.assertIn(Step(Step.Action.Skip, card_2, rotate=None, pos=None), steps_2)
        self.assertEqual(len(steps_2), 4)

        self.assertEqual(len(status.get_possible_steps()), 15)

    def test_status_steps_v2(self):
        stage = Stage(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.MySpecial],
        ]))
        card_0 = Card(np.array([
            [Grid.MyInk, Grid.MyInk],
            [Grid.MyInk, Grid.MyInk],
        ]), 10)
        card_1 = Card(np.array([
            [Grid.MyInk, Grid.MyInk, Grid.MyInk, Grid.MyInk],
        ]), 10)
        status = Status(
            stage=stage,
            hands=[card_0, card_1],
            round=12,
            my_sp=2,
            his_sp=0,
            my_deck=[],
            his_deck=[],
        )
        steps_0 = status.get_possible_steps(card_0)
        self.assertIn(Step(Step.Action.Place, card_0, rotate=0, pos=np.array([3, 2])), steps_0)
        self.assertIn(Step(Step.Action.Place, card_0, rotate=0, pos=np.array([2, 2])), steps_0)
        self.assertIn(Step(Step.Action.Place, card_0, rotate=0, pos=np.array([2, 3])), steps_0)
        self.assertIn(Step(Step.Action.Skip, card_0, rotate=None, pos=None), steps_0)

        steps_1 = status.get_possible_steps(card_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=0, pos=np.array([4, 0])), steps_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=0, pos=np.array([3, 0])), steps_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=0, pos=np.array([3, 1])), steps_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=1, pos=np.array([1, 3])), steps_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=1, pos=np.array([0, 3])), steps_1)
        self.assertIn(Step(Step.Action.Place, card_1, rotate=1, pos=np.array([0, 4])), steps_1)
        self.assertIn(Step(Step.Action.Skip, card_1, rotate=None, pos=None), steps_1)

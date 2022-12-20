import unittest

import numpy as np

from tableturf.ai.alpha.alpha import Alpha, Evaluation
from tableturf.model import Stage, Grid, Card, Status, Step


class TestAlpha(unittest.TestCase):

    def test_next_step(self):
        stage = Stage(np.array([
            [Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Empty, Grid.MySpecial, Grid.Empty],
            [Grid.Empty, Grid.Empty, Grid.Empty],
        ]))
        card_0 = Card(np.array([
            [Grid.MySpecial],
        ]), 10)
        card_1 = Card(np.array([
            [Grid.MyInk],
        ]), 20)
        card_2 = Card(np.array([
            [Grid.MySpecial],
            [Grid.MyInk],
        ]), 40)
        card_3 = Card(np.array([
            [Grid.MyInk],
            [Grid.MyInk],
        ]), 40)
        card_4 = Card(np.array([
            [Grid.MySpecial, Grid.MyInk],
            [Grid.MyInk, Grid.Empty],
        ]), 40)
        card_5 = Card(np.array([
            [Grid.MyInk, Grid.MySpecial],
            [Grid.MyInk, Grid.Empty],
        ]), 40)
        status = Status(
            stage=stage,
            hands=[card_0, card_1, card_2, card_3],
            round=3,
            my_sp=2,
            his_sp=0,
            my_deck=[card_4, card_5],
            his_deck=[],
        )

        ai = Alpha()
        print(ai.next_step(status))
        ai.reset()

    def test_evaluation_occupation(self):
        stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.HisSpecial, Grid.Wall],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Wall, Grid.Wall],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Wall, Grid.Empty, Grid.MySpecial, Grid.Empty, Grid.Wall],
            [Grid.Wall, Grid.Wall, Grid.Empty, Grid.Wall, Grid.Wall],
        ]))

        self.assertEqual(1, Evaluation.occupied_grids(stage, my_dilate=0, his_dilate=0, connectivity=8))
        self.assertEqual(2, Evaluation.occupied_grids(stage, my_dilate=0, his_dilate=0, connectivity=4))
        self.assertEqual(8, Evaluation.occupied_grids(stage, my_dilate=1, his_dilate=0, connectivity=8))
        self.assertEqual(9, Evaluation.occupied_grids(stage, my_dilate=1, his_dilate=0, connectivity=4))
        self.assertEqual(8, Evaluation.occupied_grids(stage, my_dilate=1, his_dilate=1, connectivity=8))
        self.assertEqual(9, Evaluation.occupied_grids(stage, my_dilate=1, his_dilate=1, connectivity=4))
        self.assertEqual(11, Evaluation.occupied_grids(stage, my_dilate=2, his_dilate=0, connectivity=8))
        self.assertEqual(11, Evaluation.occupied_grids(stage, my_dilate=2, his_dilate=0, connectivity=4))

        self.assertEqual(6, Evaluation.conflict_grids(stage, my_dilate=2, his_dilate=2))
        self.assertEqual(13, Evaluation.conflict_grids(stage, my_dilate=3, his_dilate=3))

    def test_evaluation_distance(self):
        stage = Stage(np.array([
            [Grid.Wall, Grid.HisInk, Grid.Empty, Grid.Empty, Grid.HisSpecial],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Wall, Grid.MySpecial, Grid.Empty, Grid.Wall, Grid.Wall],
        ]))

        self.assertEqual(4, Evaluation.square_distance(stage, np.array([4, 1])))
        self.assertEqual(1, Evaluation.square_distance(stage, np.array([0, 3])))

        stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.HisSpecial],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Wall, Grid.MySpecial, Grid.Empty, Grid.Wall, Grid.Wall],
        ]))
        self.assertEqual(5, Evaluation.square_distance(stage, np.array([4, 1])))

    def test_evaluation_ink_size(self):
        stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.MyInk],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Wall],
            [Grid.Wall, Grid.MySpecial, Grid.Empty, Grid.Wall, Grid.Wall],
        ]))

        self.assertEqual(5, Evaluation.ink_size(stage))

    def test_evaluation_compaction(self):
        stage = Stage(np.array([
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.HisSpecial, Grid.Wall],
            [Grid.Wall, Grid.Empty, Grid.Empty, Grid.Wall, Grid.Wall],
            [Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty, Grid.Empty],
            [Grid.Wall, Grid.Empty, Grid.MySpecial, Grid.Empty, Grid.Wall],
            [Grid.Wall, Grid.Wall, Grid.Empty, Grid.Wall, Grid.Wall],
        ]))

        self.assertEqual(7, Evaluation.dilated_area(stage, dilate=1))
        self.assertEqual(11, Evaluation.dilated_area(stage, dilate=2))

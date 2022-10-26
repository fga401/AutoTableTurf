import unittest

import numpy as np

from tableturf.model import Card, Grid


class TestCard(unittest.TestCase):
    def setUp(self) -> None:
        self.card = Card(
            np.array([
                [Grid.MyInk, Grid.MyInk, Grid.MyInk],
                [Grid.Empty, Grid.Empty, Grid.MySpecial],
            ]),
            5
        )

    def test_card_pattern_squares(self):
        self.assertListEqual(self.card.get_pattern(0).squares.tolist(), [
            Grid.MyInk.value, Grid.MyInk.value, Grid.MyInk.value, Grid.MySpecial.value,
        ])
        self.assertListEqual(self.card.get_pattern(1).squares.tolist(), [
            Grid.MyInk.value, Grid.MySpecial.value, Grid.MyInk.value, Grid.MyInk.value,
        ])
        self.assertListEqual(self.card.get_pattern(2).squares.tolist(), [
            Grid.MySpecial.value, Grid.MyInk.value, Grid.MyInk.value, Grid.MyInk.value,
        ])
        self.assertListEqual(self.card.get_pattern(3).squares.tolist(), [
            Grid.MyInk.value, Grid.MyInk.value, Grid.MySpecial.value, Grid.MyInk.value,
        ])

    def test_card_offset(self):
        self.assertListEqual(self.card.get_pattern(0).offset.tolist(), [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 2],
        ])
        self.assertListEqual(self.card.get_pattern(1).offset.tolist(), [
            [0, 0],
            [0, 1],
            [1, 0],
            [2, 0],
        ])
        self.assertListEqual(self.card.get_pattern(2).offset.tolist(), [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],
        ])
        self.assertListEqual(self.card.get_pattern(3).offset.tolist(), [
            [0, 0],
            [1, 0],
            [2, -1],
            [2, 0],
        ])

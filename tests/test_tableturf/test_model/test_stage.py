import unittest

import numpy as np

from tableturf.model import Stage, Grid


class TestStage(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Stage(np.array([
            [Grid.Wall.value, Grid.MySpecial.value, Grid.MyInk.value],
            [Grid.HisSpecial.value, Grid.MyInk.value, Grid.Empty.value],
            [Grid.MySpecial.value, Grid.MyInk.value, Grid.HisInk.value]
        ]))

    def test_stage_my_ink(self):
        self.assertListEqual(self.stage.my_ink.tolist(), [
            [0, 1],
            [0, 2],
            [1, 1],
            [2, 0],
            [2, 1],
        ])

    def test_stage_my_sp(self):
        self.assertListEqual(self.stage.my_sp.tolist(), [
            [0, 1],
            [2, 0],
        ])

    def test_stage_my_fiery_sp(self):
        self.assertListEqual(self.stage.my_fiery_sp.tolist(), [
            [2, 0],
        ])

    def test_stage_my_unfiery_sp(self):
        self.assertListEqual(self.stage.my_unfiery_sp.tolist(), [
            [0, 1],
        ])

    def test_stage_my_neighborhoods(self):
        self.assertListEqual(self.stage.my_neighborhoods.tolist(), [
            [1, 2],
        ])

    def test_stage_my_sp_neighborhoods(self):
        self.assertListEqual(self.stage.my_sp_neighborhoods.tolist(), [
            [0, 2],
            [1, 1],
            [1, 2],
            [2, 1],
        ])

    def test_stage_his_ink(self):
        self.assertListEqual(self.stage.his_ink.tolist(), [
            [1, 0],
            [2, 2],
        ])

    def test_stage_his_sp(self):
        self.assertListEqual(self.stage.his_sp.tolist(), [
            [1, 0],
        ])

    def test_stage_his_fiery_sp(self):
        self.assertListEqual(self.stage.his_fiery_sp.tolist(), [
            [1, 0],
        ])

    def test_stage_his_unfiery_sp(self):
        self.assertListEqual(self.stage.his_unfiery_sp.tolist(), [])

    def test_stage_his_neighborhoods(self):
        self.assertListEqual(self.stage.his_neighborhoods.tolist(), [
            [1, 2],
        ])

    def test_stage_his_sp_neighborhoods(self):
        self.assertListEqual(self.stage.his_sp_neighborhoods.tolist(), [
            [1, 1],
            [2, 1],
        ])

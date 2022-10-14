import requests
import unittest

from controller import NxbtController, Controller

endpoint = "http://192.168.50.101:5000/"


def ready() -> bool:
    resp = requests.get(endpoint)
    return resp.status_code == 200


@unittest.skipIf(not ready(), "nxbt server not ready")
class TestCard(unittest.TestCase):
    def setUp(self) -> None:
        self.client = NxbtController(endpoint)

    def test_press_buttons(self):
        self.assertTrue(self.client.press_buttons([Controller.Button.A, Controller.Button.HOME]))

    def test_tilt_stick(self):
        self.assertTrue(self.client.tilt_stick(Controller.Stick.L_STICK, 100, 0))

    def test_macro(self):
        macro = """
            A 0.2s 
            0.2s 
            HOME 0.2s 
            0.2s 
        """
        self.assertTrue(self.client.macro(macro))

import unittest

from controller import DummyController


@unittest.skipIf(False, "nxbt server not ready")
class TestCard(unittest.TestCase):
    def setUp(self) -> None:
        self.client = DummyController(block=False)

    def test_macro(self):
        macro = """
            A B 0.2s 
            A X 0.1s 
            0.2s 
            
            HOME L_STICK@-100+000 0.2s 
            0.2s 
        """
        self.assertTrue(self.client.macro(macro))

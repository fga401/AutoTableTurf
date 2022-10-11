from controller import Controller


class DummyController(Controller):
    def press_buttons(self, buttons: Controller.Button, down: float = 0.1, up: float = 0.1):
        pass

    def tilt_stick(self, stick: Controller.Stick, x: float, y: float, tilted: float = 0.1, released: float = 0.1):
        pass

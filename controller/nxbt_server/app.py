from flask import Flask, request

import nxbt

# Init NXBT
nx = nxbt.Nxbt(debug=False)
controller_idx = -1


def init_nxbt():
    global controller_idx
    # Get a list of all available Bluetooth adapters
    print('available Bluetooth adapters:')
    print(nx.get_available_adapters())
    # Get a list of all connected Nintendo Switches
    switch_addresses = nx.get_switch_addresses()
    print('connected Nintendo Switchs:')
    print(switch_addresses)
    # create controllers.
    print('creating controller...')
    controller_idx = nx.create_controller(nxbt.PRO_CONTROLLER, reconnect_address=switch_addresses)
    print('connecting...')
    # Wait for the switch to connect to the controller
    nx.wait_for_connection(controller_idx)
    print('connected.')


init_nxbt()

button_map = {
    'Y': nxbt.Buttons.Y,
    'X': nxbt.Buttons.X,
    'B': nxbt.Buttons.B,
    'A': nxbt.Buttons.A,
    'JCL_SR': nxbt.Buttons.JCL_SR,
    'JCL_SL': nxbt.Buttons.JCL_SL,
    'R': nxbt.Buttons.R,
    'ZR': nxbt.Buttons.ZR,
    'L': nxbt.Buttons.L,
    'ZL': nxbt.Buttons.ZL,
    'MINUS': nxbt.Buttons.MINUS,
    'PLUS': nxbt.Buttons.PLUS,
    'R_STICK_PRESS': nxbt.Buttons.R_STICK_PRESS,
    'L_STICK_PRESS': nxbt.Buttons.L_STICK_PRESS,
    'HOME': nxbt.Buttons.HOME,
    'CAPTURE': nxbt.Buttons.CAPTURE,
    'DPAD_DOWN': nxbt.Buttons.DPAD_DOWN,
    'DPAD_UP': nxbt.Buttons.DPAD_UP,
    'DPAD_RIGHT': nxbt.Buttons.DPAD_RIGHT,
    'DPAD_LEFT': nxbt.Buttons.DPAD_LEFT,
    'JCR_SR': nxbt.Buttons.JCR_SR,
    'JCR_SL': nxbt.Buttons.JCR_SL,
}

stick_map = {
    'L_STICK': nxbt.Sticks.LEFT_STICK,
    'R_STICK': nxbt.Sticks.RIGHT_STICK,
}

app = Flask(__name__)


@app.route('/press_buttons', methods=['POST'])
def press_buttons():
    buttons = request.args.get('buttons', default=None, type=str)
    if buttons is None:
        return
    buttons = [button_map[b] for b in buttons.upper().split(',')]
    down = request.args.get('down', default=0.3, type=float)
    up = request.args.get('up', default=0.3, type=float)
    block = request.args.get('block', default=True, type=bool)
    print('press_buttons', buttons, down, up, block)
    nx.press_buttons(controller_idx, buttons, down=down, up=up, block=block)
    return 'ok'


@app.route('/tilt_stick', methods=['POST'])
def tilt_stick():
    stick = request.args.get('stick', default=None, type=str)
    x = request.args.get('x', default=None, type=int)
    y = request.args.get('y', default=None, type=int)
    if stick is None or x is None or y is None:
        return
    stick = stick_map[stick.upper()]
    tilted = request.args.get('tilted', default=0.3, type=float)
    released = request.args.get('released', default=0.3, type=float)
    block = request.args.get('block', default=True, type=bool)
    print('tilt_stick', stick, x, y, tilted, released, block)
    nx.tilt_stick(controller_idx, stick, x=x, y=y, tilted=tilted, released=released, block=block)
    return 'ok'


@app.route('/macro', methods=['POST'])
def macro():
    macro = str(request.get_data(), 'utf-8')
    block = request.args.get('block', default=True, type=bool)
    print('macro', macro, block)
    nx.macro(controller_idx, macro, block=block)
    return 'ok'


@app.route('/')
def health_check():
    return 'ok'

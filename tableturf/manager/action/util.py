from typing import List

from controller import Controller


def buttons_to_marco(buttons: List[Controller.Button], down=0.1, up=0.1) -> str:
    return ''.join([f'{str(b.value)} {down}s\n{up}s\n' for b in buttons])

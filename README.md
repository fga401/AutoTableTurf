# AutoTableTurf

Automate the Tableturf game helping you reach Level 50 and get all sleeves. The script is based on image recognition and bluetooth emulator to auto play Tableturf.

![image](https://user-images.githubusercontent.com/36651740/194977551-2014cff7-5fe4-4964-aad9-7a467aba9aef.png)

## Features

- Simple web portal.
- AI that can beat Level 3 NPC.
- Complete flow control. Once win a NPC 30 times, auto switch to the next one.

## Getting Started

prerequisite:

- Bluetooth adapter. Tested on Raspberry 4B.
- Capture card. Tested on Razer Ripsaw HD.

> Note: all parameters about image recognition are based on Razer Ripsaw HD. It may need to finetune for other devices.

Install the requirements:

```bash
sudo pip3 install -r requirement.txt
```

Setup and run the virtual controller server on the device which has Bluetooth adapter. Please refer
to: https://github.com/fga401/AutoTableTurf/tree/master/controller/nxbt_server

Run the web portal:

```bash
export FLASK_APP=portal
sudo python3 -m flask run --host=0.0.0.0
```

On the web portal:

1. enter the virtual controller server endpoint and click `Connect`. If successful, you can control your Switch by
   keyboard.
2. Choose the correct webcam whose source is Switch.
3. Write the profile on the right side.
4. Go to the NPC selection page.
5. [Optional] Set the timer to auto stop. Also, you can check the checkbox `Turn off Switch after stop`.
6. Click 'Run'.

![image](https://user-images.githubusercontent.com/36651740/226627357-4169bf07-ee44-4739-915c-4413efcae0fe.png)

Profile example:
```json
[
  {
    "current_level": 1,
    "current_win": 2,
    "target_level": 3,
    "target_win": 30,
    "deck": 0
  },
  {
    "current_level": 3,
    "current_win": 12,
    "target_level": 3,
    "target_win": 30,
    "deck": 1
  }
]
```
Each block represents the configuration of an NPC. The above profile performs the following actions:
1. Use Deck 0 to play against the first NPC Level 1 until one win.
2. Use Deck 0 to play against the first NPC Level 2 until three wins.
3. Use Deck 0 to play against the first NPC Level 3 until thirty wins.
4. Use Deck 1 to play against the second NPC Level 3 until eighteen wins.

## Demo
1. [Splatoon3 AutoTableTurf Demo (1/2)](https://youtu.be/6ZauIWV1sGA)
2. [Splatoon3 AutoTableTurf Demo (2/2)](https://youtu.be/AXANkU0uDiA)

## Plan

- [x] Virtual controller API
- [x] Screen capturing
- [x] Screen recognition & Game flow testing
- [x] Smarter AI
- [x] User-friendly interface

## Credits

Many thanks to all below repositories:

- https://github.com/Brikwerk/nxbt for implementing a Python API.

# AutoTableTurf

Automate the Tableturf game helping you reach Level 50 and get all sleeves.

![image](https://user-images.githubusercontent.com/36651740/194977551-2014cff7-5fe4-4964-aad9-7a467aba9aef.png)

## Features

- Automated script based on image recognition.
- Simple web portal.
- AI that can beat Level 3 NPC.

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
3. Enter deck selection interface in Tableturf.
4. Choose the deck you want to use.
5. Click 'Run'.

![2022-11-28 22_16_36-](https://user-images.githubusercontent.com/36651740/204300430-a0051a0e-3617-4fca-96cc-f8c6dbd25227.png)

## Plan

- [x] Virtual controller API
- [x] Screen capturing
- [x] Screen recognition & Game flow testing
- [x] Smarter AI
- [x] ~~User-friendly interface~~

## Credits

Many thanks to all below repositories:

- https://github.com/Brikwerk/nxbt for implementing a Python API.

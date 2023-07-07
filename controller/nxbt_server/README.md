# NXBT Server

Simple HTTP server running on raspberry and invoking slim [NXBT](https://github.com/Brikwerk/nxbt) Python API to emulate
a Nentendo Pro Controller.

## Getting Started

Basically, you will need to run the flask app on a Unix-like system with bluetooth. Two platforms are tested.

### Run on Raspberry 4B

Install requirements

```bash
sudo pip3 install -r requirement.txt
```

Run the server on Raspberry 4B

```bash
# change hci0's address for better connection. refer: https://github.com/Poohl/joycontrol/issues/4
hcitool cmd 0x3f 0x001 0x66 0x55 0x44 0xCB 0x58 0x94
sudo python3 -m flask run --host=0.0.0.0
```

### Run on WSL

Running on WSL is more complicated, but it's possible.
I have successfully run it with a motherboard built-in bluetooth device. No extra USB bluetooth device is required.
The basic steps are:
1. Build your own WSL kernel that enables [USB/IP](https://github.com/dorssel/usbipd-win) protocol and bluetooth support.

2. Share the bluetooth device from Windows to WSL by USB/IP protocol.

#### Build WSL kernel

Follow this [guide](https://github.com/dorssel/usbipd-win/wiki/WSL-support#building-your-own-usbip-enabled-wsl-2-kernel) to build a WSL kernel.
You will also need to enable bluetooth support during the command `sudo make menuconfig`.
- Networking Support -> All bluetooth supports
- Add all Bluetooth Device drivers

Remember to restart WSL after above steps.

```bash
wsl --shutdown
```

#### Share bluetooth device

Follow this [guide](https://devblogs.microsoft.com/commandline/connecting-usb-devices-to-wsl/) to attach your bluetooth device to WSL.

#### Start NXBT server

Start bluetooth device.
```bash
sudo modprobe bluetooth
sudo modprobe btusb
sudo service bluetooth start
```

Install requirements and start server.

```bash
sudo apt install bluez*
sudo apt install build-essential libdbus-glib-1-dev libgirepository1.0-dev
sudo apt install python3-dev
sudo apt install pkg-config
sudo pip3 install -r requirement.txt
sudo python3 -m flask run --host=0.0.0.0
```
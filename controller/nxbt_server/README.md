# NXBT Server

Simple HTTP server running on raspberry and invoking slim [NXBT](https://github.com/Brikwerk/nxbt) Python API to emulate a Nentendo Pro Controller.

## Getting Started

Install requirements
```bash
sudo pip3 install -r requirement.txt
```

Run the server

```bash
# change hci0's address for better connection. refer: https://github.com/Poohl/joycontrol/issues/4
hcitool cmd 0x3f 0x001 0x66 0x55 0x44 0xCB 0x58 0x94
sudo python3 -m flask run --host=0.0.0.0
```

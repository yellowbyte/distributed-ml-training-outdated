from collections import namedtuple

Device = namedtuple("Device", "ip id type core")

DEVICES = [
    Device("192.168.1.8", "X30201125004859", "android mobile", 8),
    Device("192.168.1.3", "X30210313027404", "android mobile", 8),
    Device("192.168.1.4", "X30210313027413", "android mobile", 8),
    Device("192.168.1.6", "X30210313027406", "android mobile", 8),
    Device("192.168.1.2", "X30210313027410", "android mobile", 8),
    Device("192.168.1.5", "X30210313027405", "android mobile", 8),
    Device("192.168.1.7", "Tab11NEU0006997", "android tablet", 8),  # tablet
]
# workstation IP
WIP = "192.168.1.10"



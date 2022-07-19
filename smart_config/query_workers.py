'''Output memory (top) and network (ping) status of devices to workers.info

Usage: ./queryworkers.py
'''

from collections import namedtuple
from pprint import pprint

import subprocess
import re


Device = namedtuple("Device", "ip id")

DEVICES = [
    Device("192.168.0.153", "X30201125004859"),
    Device("192.168.0.198", "X30210313027404"),
    Device("192.168.0.168", "X30210313027413"),
    Device("192.168.0.181", "X30210313027406"),
    Device("192.168.0.143", "X30210313027410"),
    Device("192.168.0.139", "X30210313027405"),
#    Device("192.168.0.108",""),  # tablet   
]
# workstation IP
WIP = "192.168.0.195"

TOP_CMD = "top -b -n 1 | head -5"
PING_CMD = "ping -c 5 "

FP = r"(\d+(?:\.\d*)?|\.\d+)"

def get_adb_top(did):
    return f"adb -s {did} shell {TOP_CMD}"


def call_bash(cmd):
    return subprocess.getoutput(cmd)


def parse_top(top, device_type):
    """return available memory line"""
    if device_type == "workstation":
        start_line = "MiB Mem"
    else: 
        start_line = "Mem"

    mem_line_maybe = filter(
        lambda l:l.lstrip().startswith(start_line), 
        top.split("\n"))
    mem_line = next(mem_line_maybe, None)    
    assert mem_line is not None, "cannot find memory usage from top: "+top
    return mem_line


def mib2gb(mib):
    return mib*0.00104858


def mb2gb(mb):
    return mb*0.001


def parse_mem(top, device_type):
    """return available memory number"""
    pat = FP+r"([MG])? free"
    mem_line = parse_top(top, device_type)
    match = re.search(pat, mem_line)
    assert match is not None, (
        "mem_line does not contain available memory number: "+mem_line)
    mem_ava = float(match.group(1))
    unit = match.group(2)

    # on workstation top shows size in mib 
    # on android phones top shows size in gb,mb
    # convert to common unit: gb
    if not unit: 
        # mib 
        return mib2gb(mem_ava)
    elif unit == "M":
        return mb2gb(mem_ava)
    else:  # G
        return mem_ava


def parse_net(ping):
    """return avg RTT from ping in ms"""
    pat = r"min/avg/max/mdev = "+FP+"/"+FP+"/"
    match = re.search(pat, ping)
    assert match is not None, ( 
        "ping output does not contain stats line"+ping)
    avg = float(match.group(2))
    return avg


def main():
    result = str()

    # workstation
    top = call_bash(TOP_CMD)
    mem_ava = parse_mem(top, "workstation")
    ping = call_bash(PING_CMD+WIP)    
    latency = parse_net(ping)
    result += "workstation,"+str(mem_ava)+","+str(latency)+"\n"

    # workers
    for dev in DEVICES:
        top = call_bash(get_adb_top(dev.id))
        mem_ava = parse_mem(top, "android")
        ping = call_bash(PING_CMD+dev.ip)    
        latency = parse_net(ping)
        result += "android,"+str(mem_ava)+","+str(latency)+"\n"

    with open("workers.info", "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()

'''Output memory (top) and network (ping) status of devices to workers.info

Usage: ./queryworkers.py
'''

from utils.devices import *

from collections import namedtuple
from pprint import pprint

import subprocess
import re


TOP_CMD = "top -b -n 1 | head -5"
PING_CMD = "ping -c 5 "
#CPU_UTIL = "grep 'cpu ' /proc/stat"
#CPU_UTIL = ("grep", 'cpu ', "/proc/stat")
#CPU_UTIL_AWK = "awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage \"%\"}'"
#CPU_UTIL_AWK = ("awk", '{usage=($2+$4)*100/($2+$4+$5)} END {print usage \"%\"}')
#CPU_UTIL_ALL = "grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage \"%\"}'"

# floating point number 
FP = r"(\d+(?:\.\d*)?|\.\d+)"

def get_adb_output(did, cmd):
    output = f"adb -s {did} shell {cmd}"
    return output


#def get_adb_tuple_output(did, cmd):
#    output = ("adb", "-s", str(did), "shell") + cmd
#    return output


def call_bash(cmd):
    output = subprocess.getoutput(cmd)
    return output


#def call_bash_pipe(cmd):
#    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE)
#    output = subprocess.check_output(CPU_UTIL_AWK, stdin=ps.stdout)
#    ps.wait()
#    return output.decode("utf-8")


def parse_top(top, start_line):
    """return available memory line"""
    mem_line_maybe = filter(
        #lambda l:l.lstrip().startswith(start_line),
        #re.search("^"+start_line, top),
        lambda i: re.search("^[ ]*"+start_line, i) is not None,
        top.split("\n"))
    mem_line = next(mem_line_maybe, None)
    #breakpoint()
    try:
        assert mem_line is not None, "cannot find memory usage from top: "+top
    except Exception as e:
        breakpoint()
    return mem_line


def mib2gb(mib):
    return mib*0.00104858


def mb2gb(mb):
    return mb*0.001


def k2gb(k):
    return k/1000000


def parse_mem(top, device_type):
    """return available memory number"""
    pat = FP+r"([MGK])? free"

    start_line = str()
    if device_type == "workstation":
        start_line = "MiB Mem"
    else: 
        start_line = "Mem"
    mem_line = parse_top(top, start_line)
    match = re.search(pat, mem_line)
    try:
        assert match is not None, (
            "mem_line does not contain available memory number: "+mem_line)
    except Exception as e:
        breakpoint()
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
    elif unit == "K":
        return k2gb(mem_ava)
    else:  # G
        return mem_ava


def parse_util(top, device_type):
    """return cpu utilization number"""
    if device_type == "workstation":
        pat = FP+r"[ ]*id,"  # CPU time spent idle
        start_line = "[ ]*^\%Cpu\(s\):"
    else:
        pat = FP + r"[ ]*\%idle"  # CPU time spent idle
        start_line = FP+"[ ]*\%cpu"
    util_line = parse_top(top, start_line)
    match = re.search(pat, util_line)
    try:
        assert match is not None, (
            "util_line does not contain cpu utilization number: "+util_line)
    except Exception as e:
        print(util_line)
        breakpoint()
    cpu_util = float(match.group(1))
    return cpu_util


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
#    cpu_util_grep = call_bash(CPU_UTIL)
#   cpu_util_awk = call_bash(cpu_util_grep)
#    cpu_util_awk = call_bash_pipe(CPU_UTIL)
    mem_ava = parse_mem(top, "workstation")
#   cpu_util = parse_util(cpu_util_awk, "workstation")
    cpu_util = parse_util(top, "workstation")
    ping = call_bash(PING_CMD+WIP)
    latency = parse_net(ping)
    result += f"workstation,{str(mem_ava)},{str(cpu_util)},{str(latency)}\n"
    print("finish workstation")

    # workers
    for i,dev in enumerate(DEVICES):
        top = call_bash(get_adb_output(dev.id, TOP_CMD))
#       cpu_util_grep = call_bash(get_adb_output(dev.id, CPU_UTIL))
#       cpu_util_awk = call_bash(get_adb_output(dev.id, cpu_util_grep))
        #cpu_util_awk = call_bash_pipe(get_adb_tuple_output(dev.id, CPU_UTIL))
        #breakpoint()
        #cpu_util = call_bash(get_adb_output(dev.id, CPU_UTIL_ALL))
        mem_ava = parse_mem(top, dev.type)
#       cpu_util = parse_util(cpu_util_awk, dev.type)
        cpu_util = parse_util(top, dev.type)/dev.core
        ping = call_bash(PING_CMD+dev.ip)
        latency = parse_net(ping)
        result += f"{dev.type}{i},{str(mem_ava)},{str(cpu_util)},{str(latency)}\n"
        print("finish "+dev.id)

    with open("infiles/workers.info", "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()

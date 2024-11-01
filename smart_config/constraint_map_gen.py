"""to be run on each worker device.
output Souffle rules that encode the following constraints: MEMORY X BANDWIDTH -> LAYERS
* need at least MEMORY X BANDWIDTH to handle LAYERS *
send the constraints back to workstation
 
./constraint_map_gen.py
"""
from itertools import groupby, islice, tee
from operator import itemgetter
from typing import Tuple, List, Set,Any
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.rpc import RRef

import sys
import socket
import pickle
import signal
import os


HEADERSIZE = 10
DEVICE_ID = None

### CORE ###


def create_all_splits(layers_info: Tuple[int,int]) -> Set[Set[int]]:
    """
    Given layers encoded in `layers_info`, create
    all possible combinations

    :param layers_info: start and end index/pos of a model's layers
    :return: all combinations
    """
    def consecutive_subseq(iterable, length):
        """
        Get consecutive subsequence of size `length` in `iterable`
        source: 
        stackoverflow.com/questions/23860898/pythonic-find-all-consecutive-sub-sequences-of-certain-length
        """
        for _, consec_run in groupby(enumerate(iterable), lambda x: x[0] - x[1]):
            k_wise = tee(map(itemgetter(1), consec_run), length)
            for n, it in enumerate(k_wise):
                next(islice(it, n, n), None) # consume n items from it
            yield from zip(*k_wise)

    # setup
    start,end = layers_info
    num_layers: int = end-start+1
    layers = list(range(start,end+1))

    out: List[List[int]] = list()

    # reproduce the summation formula
    for cur_index in range(num_layers):
        # cur_index+1 number of consecutive parts                                                  
        cur_consec: int = cur_index+1
        out.extend(list(consecutive_subseq(layers, cur_consec)))
    return set(out)


def set2list(_set, _key=None, _reverse=False):
    """
    covert a set to a list ordered by `_key`. The order 
    can be reversed if `_reverse` is True
    """
    return sorted(list(_set), key=_key, reverse=_reverse)


def rand_param_gen(shape):
    """
    generate random valid input for current model portion
    using `shape`, which is extracted from `shapes`
    """
    _type = shape[2]
    if _type == "rand":
        # tensor of size shape 
        return torch.rand(shape[0],shape[1])
    elif _type == "randint":
        # tensor of size shape with high of 100
        return torch.randint(100,(shape[0],shape[1]))
#    elif _type == "randn":
#        if len(shape)
#        return torch.randn()


def getlownhigh(s: Set[int]) -> Tuple[int,Any]:
    """
    return lowest and highest number from set s
    """
    layers = set2list(s)
    start_layer = layers[0]
    if len(layers) > 1:
        end_layer = layers[-1]
    else: 
        end_layer = None
    return start_layer,end_layer


def forced_execution(s: Set[int]) -> bool:
    """
    force execute current model portion specified by `s`
    """
    start_index,end_index = getlownhigh(s)
    # perform meta-programming based on s
    init_portion, body_portion = extract_layers(start_index,end_index)
    create_model(init_portion, body_portion)
    model = MODEL()
    # execute
    inputs = rand_param_gen(SHAPES[start_index])
    try:
        with time_limit(1):
            out = model.forward(inputs)
    except TimeoutException as e:
        return False
    #breakpoint()
    return True


def forced_execution_driver() -> List[str]:
    """
    deterministically explore different portions of the model to identify 
    the model portions current MEMORY X BANDWIDTH can handle
    """
    accepted_splits = list()
    # all possible split points
    layers = set2list(SHAPES.keys())
    # all possible splits including full model
    # sorted by considering larger splits first, like full model first
    splits = set2list(create_all_splits((layers[0],layers[1])),
                _key=lambda x: len(x),
                _reverse=True)
    for s in splits: 
        #print("current split:" + str(s))
        cur_split = set(s)
        if any([cur_split.issubset(acs) for acs in accepted_splits]):
            # current split s already handled by a successful bigger split
            accepted_splits.append(cur_split)
            continue
        # perform forced execution
        success = forced_execution(cur_split)
        if success: 
            accepted_splits.append(cur_split)

    # create constraint rules from accepted_splits
    out = list()
    for acs in accepted_splits:
        start,end = getlownhigh(acs)
        if end != None:
            out.append(str(start)+","+str(end))
        else:
            out.append(str(start)+",0")
    return out


### TIME LIMITS ###


# https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


### META-PROGRAMMING ###


def extract_layers(start_index:int, end_index:int) -> Tuple[str,str]:
    """
    create the init and forward methods corresponding to the layers
    specified by start_index and end_index
    """
    # extract init_body layers
    devices_init = INIT_BODY[start_index:end_index+1]
    # extract forward_body layers
    devices_body = FORWARD_BODY[start_index:end_index+1]

    return ("\n    ".join(devices_init), "\n    ".join(devices_body))


def create_model(devices_init: str, devices_body: str) -> None:
    # create model that initialize/execute specific layers
    init_str = INIT_HEADING + [devices_init] + INIT_ENDING
    forward_str = FORWARD_HEADING + [devices_body] + FORWARD_ENDING

    # create init method
    exec("\n    ".join(init_str))
    # create forward method
    exec("\n    ".join(forward_str))
    # create model portion
    exec("MODEL = type('MODEL',(MODELBASE,),{'__init__': locals()['__init__'],'forward': locals()['forward']})")

    # make model portion visible in global scope
    exec("global MODEL")
    globals()["MODEL"] = locals()["MODEL"]


########################


def main():
    os.environ['MASTER_ADDR'] = '192.168.1.10'
    os.environ['MASTER_PORT'] = '29411'

    filepath = Path("models")
    if len(sys.argv) != 2:
        print("need to choose a model: {ff,resnet}")
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name == "ff": 
        from models.ff import (MODELBASE,INIT_BODY,FORWARD_BODY,INIT_HEADING,
            FORWARD_HEADING,INIT_ENDING,FORWARD_ENDING,SHAPES)
        globals()["SHAPES"] = SHAPES
        globals()["INIT_BODY"] = INIT_BODY
        globals()["FORWARD_BODY"] = FORWARD_BODY
        globals()["INIT_HEADING"] = INIT_HEADING
        globals()["FORWARD_HEADING"] = FORWARD_HEADING
        globals()["INIT_ENDING"] = INIT_ENDING
        globals()["FORWARD_ENDING"] = FORWARD_ENDING
        globals()["MODELBASE"] = MODELBASE
    elif model_name == "resnet":
        from models.ff import (MODELBASE,INIT_BODY,FORWARD_BODY,INIT_HEADING,
            FORWARD_HEADING,INIT_ENDING,FORWARD_ENDING,SHAPES)
    else:
        raise ValueError("unsupported model")

    # Retrieve current device name
    with open("device.txt", "r") as f:
        txt = f.read().rstrip("\n")
    global DEVICE_ID
    DEVICE_ID = txt

    # constraints is a list of model portion combination that fits the current memory X utilization
    constraints: List[str] = [DEVICE_ID]
    constraints.extend(forced_execution_driver())  # forced_execution_driver() returns list of string
    print("constraints:")
    print(constraints)
    # send `constraints` back to workstation over TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])))
    msg = pickle.dumps(constraints)
    # len(msg) is HEADERSIZE number of characters and left-aligned
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    s.send(msg)
    s.close()


if __name__ == "__main__": 
    main()
"""output Souffle rules that encode the following constraints: MEMORY X BANDWIDTH -> LAYERS
 
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
import signal


### CORE ###


def create_all_splits(layers_info: Tuple[int,int]) -> Set[List[int]]:
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
    """
    return sorted(list(_set), key=_key, reverse=_reverse)


def rand_param_gen(shape):
    """
    generate random valid input for current model portion
    using `shape`, which is extracted from `shapes`
    """
    if shape[2] == "rand":
        return torch.rand(shape[0],shape[1])
    elif shape[2] == "randint":
        return torch.randint(100,(shape[0],shape[1]))


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
    """
    start_index,end_index = getlownhigh(s)
    # perform meta-programming based on s
    init_portion, body_portion = extract_layers(start_index,end_index)
    create_model(init_portion, body_portion)
    model = MODEL()
    # execute
    inputs = rand_param_gen(SHAPES[start_index])
    try:
        with time_limit(60):
            x = model.forward(inputs)
    except TimeoutException as e:
        return False
    breakpoint()
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
        print("current split:" + str(s))
        if any([s.issubset(acs) for acs in accepted_splits]):
            # current split s already handled by a successful bigger split
            continue
        # perform forced execution
        success = forced_execution(s)
        if success: 
            accepted_splits.add(s)
    # create constraint rules from accepted_splits
    out = list()
    for acs in accepted_splits:
        start,end = getlownhigh(acs)
        if end != None:
            out.append(str(start)+','+str(end))
        else:
            out.append(str(start)+',0')
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
    init_str = INIT_HEADING + [devices_init]
    forward_str = FORWARD_HEADING + [devices_body]

    # create init method
    exec("\n    ".join(init_str))
    # create forward method
    exec("\n    ".join(forward_str))
    # create model portion
    exec("MODEL = type('MODEL',(MODELBASE,),{'__init__': locals()['__init__'],'forward': locals()['forward']})")

    # make model portion visible in global scope
    exec("global MODEL")
    globals()["MODEL"] = locals()['MODEL']


########################


def main():
    filepath = Path("models")
    if len(sys.argv) != 2:
        print("need to choose a model: {ff,resnet}")
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name == "ff": 
        from models.ff import MODELBASE,INIT_BODY,FORWARD_BODY,INIT_HEADING,FORWARD_HEADING,SHAPES
        globals()['SHAPES'] = SHAPES
        globals()['INIT_BODY'] = INIT_BODY
        globals()['FORWARD_BODY'] = FORWARD_BODY
        globals()['INIT_HEADING'] = INIT_HEADING
        globals()['FORWARD_HEADING'] = FORWARD_HEADING
        globals()['MODELBASE'] = MODELBASE
    else:
        #from models.resnet import MODELBASE,INIT_BODY,FORWARD_BODY,INIT_HEADER,FORWARD_HEADER,SHAPES
        pass

    constraints: List[str] = forced_execution_driver()    
    # TODO: send `constraints` back to workstation over TCP 


if __name__ == "__main__": 
    main() 

"""
 
./create_model_constraints.py
"""
from typing import Tuple, List, Set,Any

from utils.devices import *
from more_itertools import set_partitions  # python3.8
from copy import deepcopy

import string 
import sys
import random


def gen_random_sym(): 
    """Generate random 3 characters symbol with {A-Za-z}"""
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(3))


def flatten(seq):
    return [item for sublist in seq for item in sublist]


def get_partitions(seq,n):
    """Split seq into n unique sequential chunks"""

    # all n unique n chunks of seq
    chunks = [p for p in set_partitions(seq,n)] 
    # make sure they are sequential 
    # ex: [[2],[1,3,4]] is not sequential
    #     [[1],[2,3,4]] is sequential
    valid_chunks = filter(lambda s:flatten(s)==seq, chunks)
    return [c for c in valid_chunks]


def create_CheckUnique_rule(num_devices:int,model_portions:List):
    """
    create corresponding Souffle rule given `num_devices` and `model_portions`
    """
    # based on how many devices/workers
    syms = list()

    # CheckUnique number of arguments equals `num_devices` * 2 
    for _ in range(num_devices * 2):
        cur_sym = gen_random_sym()
        while (cur_sym in syms): 
            # get new variable name if name generated previously
            cur_sym = gen_random_sym()
        syms.append(cur_sym)

    # create function header 
    header = "CheckUnique("
    for s in syms: 
        header += s+","
    # change last char ',' to ')'
    header = header[:-1]+") :- "

    # `model_portions` affects the argument assignment
    # iterate all possible splits 
    total_constraints = str()
    cur_splits = 2            
    while (cur_splits <= len(syms)/2 and cur_splits <= len(model_portions)):
        # two-way partition, three-way partition, ...
        # ex two-way partition of [1,2,3,4,]: 
        # [[1], [2,3,4]], [[1,2], [3,4]]
        cur_partitions = get_partitions(model_portions, cur_splits)
        cur_splits += 1

        for partition in cur_partitions: 
            # we can generate a constraint for each partition
            # ex: [[1], [2,3,4]] -> a=1,b=0,c=2,d=4,e=0,f=0
            # if num_devices==3, syms = [a,b,c,d,e,f], model_portions = [1,2,3,4]
            wsyms = deepcopy(syms)
            constraint = str()
            for split in partition: 
                if len(split) == 1: 
                    start_portion = str(split[0])
                    cur_sym1 = wsyms.pop(0)
                    cur_sym2 = wsyms.pop(0)
                    constraint += cur_sym1+"="+start_portion+","+cur_sym2+"=0,"
                else: 
                    assert len(split)>1, "error in create_CheckUnique_rule"
                    start_portion = str(split[0])
                    end_portion = str(split[-1])
                    cur_sym1 = wsyms.pop(0)
                    cur_sym2 = wsyms.pop(0)
                    constraint += cur_sym1+"="+start_portion+","+cur_sym2+"="+end_portion+","
            while len(wsyms)!= 0:
                # set leftover arguments to 0
                cur_sym = wsyms.pop(0)
                constraint += cur_sym+"=0,"
            # change last , to ;
            constraint = constraint[:-1]+";"
            total_constraints += constraint
    # remove last ;
    total_constraints = total_constraints[:-1]                    
    rule = header + f"({total_constraints})."
    return rule


def create_Layers_rule():
    """
    """
    # based on memoryXbandwidth from constraint_map_generation.py
    pass


def create_SplitDevices_rule(num_devices):
    """
    create corresponding Souffle rule given `num_devices`
    """
    # based on how many devices/workers
    devices = dict()
    seen = set()

    # CheckUnique number of arguments equals `num_devices` * 2 
    for _ in range(num_devices):
        # each device has three corresponding symbols
        cur_device = list()
        for _ in range(3):
            cur_sym = None
            cur_sym = gen_random_sym()
            while (cur_sym in seen): 
                # get new variable name if name generated previously
                cur_sym = gen_random_sym()
            assert cur_sym is not None, "get_random_sym failed in create_SplitDevices_rule"
            seen.add(cur_sym)
            cur_device.append(cur_sym)
        dev_sym = cur_device.pop(0)
        devices[dev_sym] = cur_device

    # create function header 
    header = "SplitDevices("
    for s in devices.keys(): 
        header += s+","
        header += devices[s][0]+","
        header += devices[s][1]+","
    # change last char ',' to ')'
    header = header[:-1]+") :- "

    constraints = str()
    # add SplitLayer part
    for s in devices.keys():
        constraints += "SplitLayer("
        constraints += devices[s][0]+","
        constraints += devices[s][1]+","
        constraints += s+"),"

    # add 
    # A device can only be assigned one time per execution
    dev_names = list(devices.keys())
    for i,dev in enumerate(dev_names):
        for other_dev in dev_names[i+1:]:
            constraints += dev+"!="+other_dev+","

    # add CheckUnique
    constraints += "CheckUnique("
    dev_layers = list(devices.values())
    dev_layers = flatten(dev_layers)
    for layer in dev_layers: 
        constraints += layer+","
    constraints = constraints[:-1]+")."

    return header + constraints 


def write_souffle_code(rules):
    """
    """
    with open("split.dl", "w") as s, \
            open("split_base.dl", "r") as b:      
        source = b.readlines()
        source.extend(rules)
        s.write("\n".join(rules))


def main(): 
    if len(sys.argv) != 2:
        print("need to choose a model: {ff,resnet}")
        sys.exit(1)
    model_name = sys.argv[1]
    if model_name == "ff": 
        from models.ff import FORWARD_BODY
        globals()["FORWARD_BODY"] = FORWARD_BODY
    elif model_name == "resnet":
        from models.ff import (FORWARD_BODY)
        globals()["FORWARD_BODY"] = FORWARD_BODY
    else:
        raise ValueError("unsupported model")
    num_model_portions = len(FORWARD_BODY)
    model_portions = [i+1 for i in range(num_model_portions)]

    num_devices = len(DEVICES) + 1  # + 1 for workstation
    rules = list()
    rules.append(create_CheckUnique_rule(num_devices,model_portions))
    rules.append(create_SplitDevices_rule(num_devices))
    write_souffle_code(rules)


if __name__ == "__main__": 
    main() 

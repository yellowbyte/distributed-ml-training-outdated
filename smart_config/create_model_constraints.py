"""
 
./create_model_constraints.py
"""

from utils.devices import *


def gen_random_sym(): 
    


def create_CheckUnique_rule(num_devices):
    # based on how many devices/workers
    syms = set()
    pass


def create_Layers_rule():
    # based on memoryXbandwidth from constraint_map_generation.py
    pass


def create_SplitDevices_rule(num_devices):
    # based on how many devices/workers
    pass


def main(): 
    num_devices = len(DEVICES) + 1  # + 1 for workstation
    CheckUnique_rule = create_CheckUnique_rule(num_devices)
    SplitDevices_rule = create_SplitDevices_rule(num_devices)


if __name__ == "__main__": 
    main() 

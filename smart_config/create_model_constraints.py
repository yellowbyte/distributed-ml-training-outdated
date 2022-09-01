"""
 
./create_model_constraints.py
"""

from utils.devices import *

import string 
import random


def gen_random_sym(): 
    """Generate random 3 characters symbol with {A-Za-z}"""
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(3))


def create_CheckUnique_rule(num_devices,model_portions):
    """
    create corresponding Souffle rule given `num_devices` and `model_portions`
    """
    # based on how many devices/workers
    syms = set()


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
    pass


def main(): 
    num_devices = len(DEVICES) + 1  # + 1 for workstation
    create_CheckUnique_rule(num_devices)
    create_SplitDevices_rule(num_devices)


if __name__ == "__main__": 
    main() 

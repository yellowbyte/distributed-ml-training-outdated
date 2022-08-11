from itertools import groupby, islice, tee
from operator import itemgetter
from typing import Tuple, List, Set


def create_all_splits(layers_info: Tuple[int,int]) -> Set[List[int]]:
    """
    Given layers encoded in `layers_info`, create
    all possible combinations

    :param layers_info: start and end index/pos of a model's layers
    :return: all combinations
    """
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

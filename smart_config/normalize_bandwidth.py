"""Reverse values in memory.bandwidth so that smallest latency has the largest value, and so on

./normalize_latency.py memory.bandwidth
"""

import os


def main():
    input_file = output_file = "workers.bandwidth"
    workers_info = list() 
    assert os.path.isfile(input_file), "workers.bandwidth does not exist"
    with open(input_file, "r") as f:    
        workers_bandwidth = list(
            map(lambda l:l.rstrip(), f.readlines()))

    new_workers_bandwidth = list()
    # make smaller number larger
    for wbandwidth in workers_bandwidth:
        device,bandwidth = wbandwidth.split(",")
        new_bandwidth = 1/float(bandwidth)
        new_worker_bandwidth = f"{device},{new_bandwidth}"             
        new_workers_bandwidth.append(new_worker_bandwidth)
    
    with open(output_file, "w") as f:
        for wbandwidth in new_workers_bandwidth:
            f.write(wbandwidth+"\n")


if __name__ == "__main__": 
    main()

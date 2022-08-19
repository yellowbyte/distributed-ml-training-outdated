"""create Souffle input files (workers.device, workers.bandwidth, workers.memory) from workers.info.
workers.info is created by query_workers.py

./create_souffle_inputs.py
"""

import os


def main():
    input_file = "workers.info"
    workers_info = list() 
    assert os.path.isfile(input_file), "workers.info does not exist"
    with open(input_file, "r") as f:    
        workers_info = list(
            map(lambda l:l.rstrip(), f.readlines()))
    
    with open("workers.device", "w") as d, \
            open("workers.bandwidth", "w") as b, \
            open("workers.memory", "w") as m:      
        for winfo in workers_info:
            device,memory,latency = winfo.split(",")
            d.write(device+os.linesep)
            b.write(device+","+latency+os.linesep)            
            m.write(device+","+memory+os.linesep)


if __name__ == "__main__": 
    main()

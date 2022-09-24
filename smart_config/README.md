Perform Smart Splitting
-
### notes
* contains code to query each device for constraints
* constraints solving with Souffle
  * solving constraints -> output split configuration 
* Gurobi

### order of operations
1. (WS) query\_workers.py to create workers.info
  * account for CPU utilization: 
2. (WS) create\_souffle\_inputs.py to create workers.{memory,device,util,bandwidth} for split.dl from workers.info 
3. (WS) normalize\_bandwidth.py so higher latency will be assigned smaller value (update workers.bandwidth)
4. (D)  constraint\_map\_gen.py and constraint\_map\_gen\_base.py to create model constraints (layers -> current Memory X Bandwidth) for split.dl
  * currently have layers -> Memory
  * TODO: bandwidth cannot be tested individually. Add it to constraint\_map\_gen.py 
  * want: layers -> current Memory X Bandwidth X Utilization X Device 
5. (WS) from outputs from all devices, create model constraints for memory X utilization X bandwidth X device
6. (WS) TODO: create\_model\_constraints.py to create souffle rules on model portions constraints
7. (WS) split.dl to get model portion assignment
8. (WS) ASK: linear programming with Gurobi

### run souffle
* `souffle -F./infiles -D. split.dl`

### Transformer 
Expanded TransformerEncoderLayer 

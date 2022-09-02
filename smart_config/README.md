Perform Smart Splitting
-
### notes
* contains code to query each device for constraints
* constraints solving with Souffle
  * sovling constraints -> output split configuration 
* Gurobi

### order of operations
1. (WS) query\_workers.py to create workers.info
2. (WS) create\_souffle\_inputs.py to create workers.{memory,device,bandwidth} for split.dl
3. (WS) normalize\_bandwidth.py so higher latency will be assigned smaller value
4. (D)  constraint\_map\_gen.py to create model constraints (layers -> current Memory X Bandwidth) for split.dl
  * currently have layers -> Memory
  * TODO: bandwidth cannot be tested individually
5. (WS) TODO: create\_model\_constraints.py to create souffle rules on model portions constraints
6. (WS) split.dl to get model portion assignment
7. (WS) TODO: linear programming with Gurobi

### run souffle
* `souffle -F./infiles -D. split.dl`

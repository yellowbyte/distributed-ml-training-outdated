Perform Smart Splitting
-
* contains code to query each device for constraints
* constraints solving with Souffle
  * sovling constraints -> output split configuration 
* TODO: Gurobi

1. (WS) query\_workers.py to create workers.info
2. (WS) create\_souffle\_inputs.py to create workers.{memory,device,bandwidth} for split.dl
3. (WS) normalize\_bandwidth.py so higher latency will be assigned smaller value
4. (D)  constraint\_map\_gen.py to create model constraints (layers -> current Memory X Bandwidth) for split.dl
5. (WS) create\_model\_constraints.py to create souffle rules on model portions constraints
6. (WS) split.dl to get model portion assignment

RUN: 
* `souffle -F./infiles -D. split.dl`

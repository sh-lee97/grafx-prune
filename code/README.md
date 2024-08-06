# Implementation of "Searching For Music Mixing Graphs: A Pruning Apprach"

This repository contains codes for the paper named above.

``` bash
├── requirements.txt # required libraries
├── auraloss_opt # Christian Steinmetz's auraloss, but optimized a bit for speed
├── loss.py # loss function (basically uses above)
├── configs # configs for experiments conducted in the paper 
├── data # scripts and configurations for loading Medley and Mixing Secrets songs
├── mixing_console.py # generates mixing console graph 
├── prune.py # some pruning operations
├── solver.py # main solver
├── train.py # entry point; our main training script
└── utils.py # some utility functions
```

Our code uses our companion work `GRAFX`; please refer to [this repository](https://github.com/sh-lee97/grafx)
for more details.

---


To obtain a graph for a single song, e.g., `TablaBreakbeatScience_Rocksteady` from `MedleyDB`, run the following script:

``` bash
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    config=prune_hybrid_1e_2 \
    dataset=medley \
    song=TablaBreakbeatScience_RockSteady
```

To simply train a full mixing console, you can pass `config=mixing_console_full` instead.
All the arguments can be overriden from the cli. For more details, see `configs/base.yaml`.

---
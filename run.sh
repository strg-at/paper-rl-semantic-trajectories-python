#!/bin/bash

source ./activate.sh
python experiments/ppo_levenshtein.py --num-trajectories 16
python experiments/ppo_levenshtein.py --num-trajectories 256

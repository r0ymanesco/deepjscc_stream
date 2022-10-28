#!/usr/bin/env bash

python -u main.py \
    --trainer 'vct_bandwidth_allocation' \
    --config_file 'configs/vct_bandwidth_allocation.yaml' \
    --device 'cuda:1' \
    --comments 'ELIC+lossMod' \
    --resume \

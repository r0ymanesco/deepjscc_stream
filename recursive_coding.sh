#!/usr/bin/env bash

python -u main.py \
    --trainer 'recursive_coding' \
    --config_file 'configs/recursive_coding.yaml' \
    --device 'cuda:0' \
    --resume \
    --comments 'with mask'

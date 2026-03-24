#!/bin/bash

if [ $# -eq 1 ]; then
    python train.py --config snn/configs/event_snn_lite.yaml --name $1
elif [ $# -eq 2 ]; then
    python train.py --config snn/configs/event_snn_lite.yaml --name $1 --resume $2
fi
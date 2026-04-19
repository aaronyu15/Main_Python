#!/bin/bash
set -e

CONFIG="snn/configs/event_snn_lite.yaml"

# Defaults
#sed -i 's/num_bins: [0-9]\+/num_bins: 5/g' $CONFIG
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 5000/g' $CONFIG
#sed -i 's/base_ch: [0-9]\+/base_ch: 8/g' $CONFIG

## 08 - 5 bins
#NAME="08_2000u_5bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 2000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"
#
#NAME="08_5000u_5bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 5000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"
#
#NAME="08_8000u_5bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 8000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"
#
#sed -i 's/num_bins: [0-9]\+/num_bins: 10/g' $CONFIG
## 09 - 10 bins
#NAME="09_2000u_10bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 2000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"
#
#NAME="09_5000u_10bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 5000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"

#NAME="09_8000u_10bin"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 8000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"

# 06 - 16 channels

#sed -i 's/num_bins: [0-9]\+/num_bins: 5/g' $CONFIG
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 5000/g' $CONFIG
#sed -i 's/base_ch: [0-9]\+/base_ch: 16/g' $CONFIG

#NAME="07_incr_upscale"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_1"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_3"
#python train.py --config snn/configs/event_snn_lite.yaml --name "${NAME}_4"

#NAME="10_mvsec"
#sed -i 's/num_bins: [0-9]\+/num_bins: 5/g' $CONFIG
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 5000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite_mvsec.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite_mvsec.yaml --name "${NAME}_3"
#
#NAME="10_mvsec_5bin_10ms_dt1"
#sed -i 's/bin_interval_us: [0-9]\+/bin_interval_us: 10000/g' $CONFIG
#python train.py --config snn/configs/event_snn_lite_mvsec.yaml --name "${NAME}_1"
#python train.py --config snn/configs/event_snn_lite_mvsec.yaml --name "${NAME}_2"
#python train.py --config snn/configs/event_snn_lite_mvsec.yaml --name "${NAME}_3"

NAME="11_dsec_5bin_20ms"
python train.py --config snn/configs/event_snn_lite_dsec.yaml --name "${NAME}_1"
python train.py --config snn/configs/event_snn_lite_dsec.yaml --name "${NAME}_2"
python train.py --config snn/configs/event_snn_lite_dsec.yaml --name "${NAME}_3"
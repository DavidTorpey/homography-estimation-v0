#!/bin/bash
#SBATCH --job-name=cifar100-resnet50-simclr
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr/cifar100/log.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr/cifar100/log.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=bigbatch

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.train --config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr/cifar100/config.yaml

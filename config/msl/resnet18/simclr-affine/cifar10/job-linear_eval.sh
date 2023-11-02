#!/bin/bash
#SBATCH --job-name=cifar10-resnet18-simclr-affine-linear_eval
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet18/simclr-affine/cifar10/log-linear_eval.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet18/simclr-affine/cifar10/log-linear_eval.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=bigbatch

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.linear_eval.run --config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet18/simclr-affine/cifar10/config.yaml --model_path /home-mscluster/dtorpey/code/homography-estimation-v0/results/resnet18/simclr-affine/stl10/80_model_stl10.pth

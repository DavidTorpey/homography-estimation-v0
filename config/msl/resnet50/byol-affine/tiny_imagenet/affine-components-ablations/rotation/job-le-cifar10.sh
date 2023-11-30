#!/bin/bash
#SBATCH --job-name=cifar10-resnet50-byol-affine-linear_eval
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/byol-affine/tiny_imagenet/affine-components-ablations/rotation/log-linear_eval.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/byol-affine/tiny_imagenet/affine-components-ablations/rotation/log-linear_eval.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=stampede

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.linear_eval.run \
--config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/byol-affine/tiny_imagenet/affine-components-ablations/rotation/config-le-cifar10.yaml \
--model_path /home-mscluster/dtorpey/code/homography-estimation-v0/results/resnet50/byol-affine/tiny_imagenet/affine-components-ablations/rotation/90_model_tiny_imagenet.pth

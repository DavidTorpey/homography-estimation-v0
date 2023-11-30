#!/bin/bash
#SBATCH --job-name=food101-resnet50-simclr-affine-linear_eval
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr-affine/tiny_imagenet/affine-components-ablations/rotation-translation/log-linear_eval.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr-affine/tiny_imagenet/affine-components-ablations/rotation-translation/log-linear_eval.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=stampede

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.linear_eval.run \
--config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/simclr-affine/tiny_imagenet/affine-components-ablations/rotation-translation/config-le-food101.yaml \
--model_path /home-mscluster/dtorpey/code/homography-estimation-v0/results/resnet50/simclr-affine/tiny_imagenet/affine-components-ablations/rotation-translation/90_model_tiny_imagenet.pth

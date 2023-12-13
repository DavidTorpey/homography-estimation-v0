#!/bin/bash
#SBATCH --job-name=tiny_imagenet-resnet50-barlow_twins-affine-investigate-losses
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine-investigate-losses/tiny_imagenet/affine-components-ablations/rotation-scale-shear-translation/log.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine-investigate-losses/tiny_imagenet/affine-components-ablations/rotation-scale-shear-translation/log.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=bigbatch

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.train \
--config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine-investigate-losses/tiny_imagenet/affine-components-ablations/rotation-scale-shear-translation/config.yaml

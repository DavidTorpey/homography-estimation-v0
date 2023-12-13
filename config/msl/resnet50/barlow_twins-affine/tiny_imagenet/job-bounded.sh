#!/bin/bash
#SBATCH --job-name=tiny_imagenet-resnet50-barlow_twins-affine-bounded
#SBATCH --output=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine/tiny_imagenet/log.out
#SBATCH --error=/home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine/tiny_imagenet/log.err
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --partition=bigbatch

cd /home-mscluster/dtorpey/code/homography-estimation-v0

. /home-mscluster/dtorpey/code/object-centricity-ssl/env.sh

python -m he.train \
--config_path /home-mscluster/dtorpey/code/homography-estimation-v0/config/msl/resnet50/barlow_twins-affine/tiny_imagenet/config-bounded.yaml

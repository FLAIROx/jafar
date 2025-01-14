#!/bin/bash
gpu=$1
script_and_args="${@:2}"
WANDB_API_KEY=$(cat ./docker/wandb_key)
git pull

echo "Launching container jafar_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/jafar \
    --name jafar\_$gpu \
    --user $(id -u) \
    --rm \
    -d \
    jafar \
    /bin/bash -c "$script_and_args"

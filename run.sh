#!/bin/bash

#echo "TODO: fill in the docker run command"

docker run --rm \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -p 5000:5000 \
    ift6758-serving
#!/bin/bash

#echo "TODO: fill in the docker run command"

docker run --rm \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -p 8000:8000 \
    ift6758-serving
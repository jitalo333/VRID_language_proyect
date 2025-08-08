#!/bin/bash

echo "Starting TeCla-NLP container with host networking..."
echo "This will give the container direct access to Ollama on the host."

docker run \
    --rm \
    -it \
    --name teclanlp \
    --network host \
    -e OLLAMA_BASE_URL=http://localhost:11434 \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    core
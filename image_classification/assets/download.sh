#!/bin/bash
MODEL_NAME=PP-LCNet_x1_0
MODEL_LIST="PP-LCNet_x1_0 MobileNetV3_small_x1_0 PP-LCNet_x1_0_gpu MobileNetV3_small_x1_0_gpu"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported model: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the support list. Exiting."
  exit 1
fi

MODEL_URL="https://paddlelite-demo.bj.bcebos.com/paddle-x/image_classification/models/${MODEL_NAME}.tar.gz"

MODELS_DIR="$(pwd)/models/"

if [ ! -d "$(pwd)/models" ]; then
  mkdir $(pwd)/models
fi

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -zxvf download.tar.gz
  rm -f download.tar.gz
  
  cd ..
}

download_and_uncompress "${MODEL_URL}" "${MODELS_DIR}"

echo "Download successful!"

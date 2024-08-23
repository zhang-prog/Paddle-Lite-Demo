#!/bin/bash
MODEL_NAME=PicoDet-S
MODEL_LIST="PicoDet-S PicoDet-L PicoDet_layout_1x"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported model: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the support list. Exiting."
  exit 1
fi

MODEL_URL="https://paddlelite-demo.bj.bcebos.com/paddle-x/object_detection/models/${MODEL_NAME}.tar.gz"
MODELS_DIR="$(pwd)/models/"
# dev cpu + gpu lib(fix picode run error)
ANDROID_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/android/paddle_lite_libs_dev_gpu.tar.gz"
# dev cpu lib(fix picode run error)
IOS_LIBS_URL="https://paddlelite-demo.bj.bcebos.com/libs/ios/paddle_lite_libs_dev.tar.gz"

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
ANDROID_DIR="$(pwd)/../../libs/android"
IOS_DIR="$(pwd)/../../libs/ios"
download_and_uncompress "${ANDROID_LIBS_URL}" "${ANDROID_DIR}"
download_and_uncompress "${IOS_LIBS_URL}" "${IOS_DIR}"

echo "Download successful!"

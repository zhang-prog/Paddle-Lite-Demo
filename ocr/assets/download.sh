#!/bin/bash
MODEL_NAME=PP-OCRv4_mobile
MODEL_LIST="PP-OCRv3_mobile PP-OCRv4_mobile"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported model: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the support list. Exiting."
  exit 1
fi

IMAGES_URL="https://paddlelite-demo.bj.bcebos.com/demo/ocr/images/images.tar.gz"
LABELS_URL="https://paddlelite-demo.bj.bcebos.com/demo/ocr/labels/labels.tar.gz"
CLS_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/demo/ocr/models/ch_ppocr_mobile_v2.0_cls_slim_opt_for_cpu_v2_10_rc.tar.gz"
DET_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/paddle-x/ocr/models/${MODEL_NAME}_det.tar.gz"
REC_MODEL_URL="https://paddlelite-demo.bj.bcebos.com/paddle-x/ocr/models/${MODEL_NAME}_rec.tar.gz"
CONFIG_TXT_URL="https://paddlelite-demo.bj.bcebos.com/demo/ocr/config.tar.gz"
MODELS_DIR="$(pwd)/models/"
IMAGES_DIR="$(pwd)/images/"
LABELS_DIR="$(pwd)/labels/"

if [ ! -d "$(pwd)/models" ]; then
  mkdir $(pwd)/models
fi
if [ ! -d "$(pwd)/images" ]; then
  mkdir $(pwd)/images
fi
if [ ! -d "$(pwd)/labels" ]; then
  mkdir $(pwd)/labels
fi

download_and_uncompress() {
  local url="$1"
  local dir="$2"
  
  echo "Start downloading ${url}"
  curl -L ${url} > ${dir}/download.tar.gz
  cd ${dir}
  tar -xvf download.tar.gz
  rm -f download.tar.gz
  
  cd ..
}

download_and_uncompress "${CLS_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${DET_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${REC_MODEL_URL}" "${MODELS_DIR}"
download_and_uncompress "${IMAGES_URL}" "${IMAGES_DIR}"
download_and_uncompress "${LABELS_URL}" "${LABELS_DIR}"

echo "Download successful!"

#!/bin/bash
MODEL_NAME=PP-OCRv4_mobile
ASSETS_DIR="$(pwd)/../../../assets"
MODEL_LIST="PP-OCRv3_mobile PP-OCRv4_mobile"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported models: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the supported models. Now exiting."
  exit 1
fi

if [ ! -f "${ASSETS_DIR}/models/${MODEL_NAME}_det.nb" ];then
  echo "Model ${MODEL_NAME}_det not found! "
  exit 1
fi

if [ ! -f "${ASSETS_DIR}/models/${MODEL_NAME}_rec.nb" ];then
  echo "Model ${MODEL_NAME}_rec not found! "
  exit 1
fi

# push
adb push ./ppocr_demo /data/local/tmp/
ppocr_demo_path="/data/local/tmp/ppocr_demo"

# run
adb shell "cd ${ppocr_demo_path} \
           && chmod +x ./ppocr_demo \
           && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ppocr_demo \
                \"./models/${MODEL_NAME}_det.nb\" \
                \"./models/${MODEL_NAME}_rec.nb\" \
                ./models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/ppocr_keys_v1.txt \
                ./config.txt"

adb pull ${ppocr_demo_path}/test_img_result.jpg .

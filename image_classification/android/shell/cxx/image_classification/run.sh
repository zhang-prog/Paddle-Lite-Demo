#!/bin/bash
MODEL_NAME=PP-LCNet_x1_0
MODEL_LIST="PP-LCNet_x1_0 MobileNetV3_small_x1_0"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported model: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the support list. Exiting."
  exit 1
fi

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/image_classify"
ARM_ABI=arm64-v8a # arm64-v8a or armeabi-v7a

if [ -n "$2" ]; then
  ARM_ABI="$2"
fi

if [ ! -d "${ASSETS_DIR}/models/${MODEL_NAME}" ];then
  echo "Model $MODEL_NAME not found! "
  exit 1
fi

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "cd /data/local/tmp/ && mkdir image_classify"
# push
adb push ./build/image_classification ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
echo "--run model on cpu---"
adb shell "cd ${ADB_DIR} \
           && chmod +x ./image_classification \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./image_classification \
               ./models/${MODEL_NAME}/model.nb \
               ./images/tabby_cat.jpg \
               ./labels/labels.txt \
               3 224 224 \
               0 1 10 1 0 \
               "

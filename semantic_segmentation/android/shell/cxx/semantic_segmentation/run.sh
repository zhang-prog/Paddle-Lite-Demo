#!/bin/bash
MODEL_NAME=PP-LiteSeg-T
MODEL_LIST="PP-LiteSeg-T"

if [ -n "$1" ]; then
  MODEL_NAME="$1"
fi

if ! echo "$MODEL_LIST" | grep -qw "$MODEL_NAME"; then
  echo "Supported model: ${MODEL_LIST}"
  echo "$MODEL_NAME is not in the support list. Exiting."
  exit 1
fi

PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/semantic_segmentation"
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
adb shell "rm -rf ${ADB_DIR}"
adb shell "cd /data/local/tmp/ && mkdir semantic_segmentation"
# push
adb push ./build/semantic_segmentation ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
adb shell "cd ${ADB_DIR} \
           && chmod +x ./semantic_segmentation \
           && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
           &&  ./semantic_segmentation \
               ./models/${MODEL_NAME}/model.nb    \
               ./images/test.jpg   \
               ./labels/label_list  \
               1024 512 4    \
               0 10 10 0    \
               "
adb pull ${ADB_DIR}/result.jpg ./
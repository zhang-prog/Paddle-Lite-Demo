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

PADDLE_LITE_DIR="$(pwd)/../../../../../libs/android/cxx"
OPENCV_LITE_DIR="$(pwd)/../../../../../libs/android/opencv4.1.0"
ASSETS_DIR="$(pwd)/../../../../assets"
ADB_DIR="/data/local/tmp/picodet_detection"
ARM_ABI=arm64-v8a # arm64-v8a or armeabi-v7a

if [ ! -d "${ASSETS_DIR}/models/${MODEL_NAME}" ];then
  echo "Model ${MODEL_NAME} not found! "
  exit 1
fi

if [ -n "$2" ]; then
  ARM_ABI="$2"
fi

echo "PADDLE_LITE_DIR is ${PADDLE_LITE_DIR}"
echo "OPENCV_LITE_DIR is ${OPENCV_LITE_DIR}"
echo "ASSETS_DIR is ${ASSETS_DIR}"
echo "ADB_DIR is ${ADB_DIR}"
# mkdir
adb shell "cd /data/local/tmp/ && mkdir picodet_detection"
# push
adb push ./build/picodet_detection ${ADB_DIR}
adb push ${ASSETS_DIR}/models/ ${ADB_DIR}
adb push ${ASSETS_DIR}/images/ ${ADB_DIR}
adb push ${ASSETS_DIR}/labels/ ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libc++_shared.so  ${ADB_DIR}
adb push ${PADDLE_LITE_DIR}/libs/${ARM_ABI}/libpaddle_light_api_shared.so  ${ADB_DIR}

# run
if [ "$MODEL_NAME" == "PicoDet-S" -o "$MODEL_NAME" == "PicoDet-L" ]; then
  adb shell "cd ${ADB_DIR} \
            && chmod +x ./picodet_detection \
            && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
            &&  ./picodet_detection \
                ./models/${MODEL_NAME}/model.nb \
                ./images/dog.jpg \
                ./labels/coco_label_list.txt \
                0.5 320 320 \
                0 1 10 1 0 \
            "
  adb pull ${ADB_DIR}/dog_picodet_detection_result.jpg ./
fi 

if [ "$MODEL_NAME" == "PicoDet_layout_1x" ]; then
  adb shell "cd ${ADB_DIR} \
            && chmod +x ./picodet_detection \
            && export LD_LIBRARY_PATH=${ADB_DIR}:${LD_LIBRARY_PATH} \
            &&  ./picodet_detection \
                ./models/${MODEL_NAME}/model.nb \
                ./images/paper.jpg \
                ./labels/picodet_layout_label_list.txt \
                0.5 608 800 \
                0 1 10 1 0 \
            "
  adb pull ${ADB_DIR}/paper_picodet_detection_result.jpg ./
fi

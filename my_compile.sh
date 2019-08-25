#!/usr/bin/env bash

COBJECTFLOW_EXTERNAL_PATH=$PWD/../cobjectflow-external
#COBJECTFLOW_EXTERNAL_PATH=/opt/matrix_dependencies
BUILD_DIR=$1
BUILD_TYPE=$2
COMPILE_THREAD_NUM=$3

echo "param 0: $0"
echo "BUILD_DIR param 1: $1"
echo "BUILD_TYPE param 2: $2"
echo "compile with $COMPILE_THREAD_NUM threads."
echo "Start build CObjectFlow..."
if [ -d "$BUILD_DIR" ];
    then echo "Remove the previous building files..."
    rm -r $BUILD_DIR
fi
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
-DPROTOBUF_PATH=$COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1/install \
-DOPENCV_PATH=$COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10/install \
-DCAFFE_PATH=$COBJECTFLOW_EXTERNAL_PATH/caffe/install \
-DJSON_PATH=$COBJECTFLOW_EXTERNAL_PATH/jsoncpp-1.8.4/install \
-DCNPY_PATH=$COBJECTFLOW_EXTERNAL_PATH/cnpy/install
make -j $COMPILE_THREAD_NUM
echo "Build DetectionEngine done."


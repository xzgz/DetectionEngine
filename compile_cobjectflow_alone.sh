#!/usr/bin/env bash

COBJECTFLOW_EXTERNAL_PATH=$PWD/../cobjectflow-external
#COBJECTFLOW_EXTERNAL_PATH=/opt/matrix_dependencies

echo "Start build CObjectFlow..."
if [ -d "build" ];
    then echo "Remove the previous building files..."
    rm -r build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
-DCC=/usr/local/gcc-4.9.4/bin/gcc \
-DPROTOBUF_PATH=$COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1/install \
-DOPENCV_PATH=$COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10/install \
-DCAFFE_PATH=$COBJECTFLOW_EXTERNAL_PATH/caffe/install \
-DJSON_PATH=$COBJECTFLOW_EXTERNAL_PATH/jsoncpp-1.8.4/install \
-DCNPY_PATH=$COBJECTFLOW_EXTERNAL_PATH/cnpy/install
make -j 8
echo "Build CObjectFlow done."



#!/usr/bin/env bash

COMPILE_THREAD_NUM=$1
echo "compile with $COMPILE_THREAD_NUM threads."

cd ..
#git clone git@gitlab.bj.sensetime.com:heyanguang/cobjectflow-external.git
COBJECTFLOW_EXTERNAL_PATH=$PWD/cobjectflow-external
#COBJECTFLOW_EXTERNAL_PATH=/opt/matrix_dependencies

cd $COBJECTFLOW_EXTERNAL_PATH/boost_1_70_0
echo "Start build boost-1.70.0..."
if [ -d "install" ];
    then echo "Remove the previous install files..."
    rm -r install
fi
./bootstrap.sh --with-libraries=all --with-toolset=gcc
./b2 toolset=gcc -j $COMPILE_THREAD_NUM
./b2 install --prefix=$COBJECTFLOW_EXTERNAL_PATH/boost_1_70_0/install
cd $COBJECTFLOW_EXTERNAL_PATH/boost_1_70_0/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build boost-1.70.0 done."

cd $COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1
echo "Start build protobuf-3.6.1..."
if [ -d "install" ];
    then echo "Remove the previous install files..."
    rm -r install
fi
./autogen.sh
make clean
./configure --prefix=$COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1/install
make -j $COMPILE_THREAD_NUM
make install
cd $COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build protobuf-3.6.1 done."

cd $COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10
echo "Start build opencv-2.4.10..."
if [ -d "build" ];
    then echo "Remove the previous building files..."
    rm -r build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10/install \
-DWITH_CUDA=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TIFF=ON -DBUILD_DOCS=OFF \
-DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON -DBUILD_TESTS=OFF
make -j $COMPILE_THREAD_NUM
make install
cd $COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build opencv-2.4.10 done."

cd $COBJECTFLOW_EXTERNAL_PATH/caffe
echo "Start build caffe..."
if [ -d "build" ];
    then echo "Remove the previous building files..."
    rm -r build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$COBJECTFLOW_EXTERNAL_PATH/caffe/install \
-DPROTOBUF_PATH=$COBJECTFLOW_EXTERNAL_PATH/protobuf-3.6.1/install \
-DOPENCV_PATH=$COBJECTFLOW_EXTERNAL_PATH/opencv-2.4.10/install \
-DBOOST_PATH=$COBJECTFLOW_EXTERNAL_PATH/boost_1_70_0/install
make -j $COMPILE_THREAD_NUM
make install
cd $COBJECTFLOW_EXTERNAL_PATH/caffe/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build caffe done."

cd $COBJECTFLOW_EXTERNAL_PATH/cnpy
echo "Start build cnpy..."
if [ -d "build" ];
    then echo "Remove the previous building files..."
    rm -r build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$COBJECTFLOW_EXTERNAL_PATH/cnpy/install \
-DENABLE_STATIC=OFF
make -j $COMPILE_THREAD_NUM
make install
cd $COBJECTFLOW_EXTERNAL_PATH/cnpy/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build cnpy done."

cd $COBJECTFLOW_EXTERNAL_PATH/jsoncpp-1.8.4
echo "Start build jsoncpp-1.8.4..."
if [ -d "build" ];
    then echo "Remove the previous building files..."
    rm -r build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$COBJECTFLOW_EXTERNAL_PATH/jsoncpp-1.8.4/install \
-DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF
make -j $COMPILE_THREAD_NUM
make install
cd $COBJECTFLOW_EXTERNAL_PATH/jsoncpp-1.8.4/install
if [ ! -d "lib" ];
    then if [ -d "lib64" ];
        then ln -s lib64 lib
    fi
fi
cd -
echo "Build jsoncpp-1.8.4 done."

cd $COBJECTFLOW_EXTERNAL_PATH/../DetectionEngine
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
make -j $COMPILE_THREAD_NUM
echo "Build DetectionEngine done."


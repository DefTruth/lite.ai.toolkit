#!/bin/bash

BUILD_DIR=build

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
  echo "creating build dir: ${BUILD_DIR} ..."
else
  echo "build dir: ${BUILD_DIR} directory exist! ..."
fi

cd "${BUILD_DIR}" && pwd 
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_INSTALL_PREFIX=./install
make -j8
make install

#!/bin/bash
if [ ! -d ./build ]; then
  mkdir build
else
  echo "build directory exist! clearing ..."
  rm -rf ./build/* && echo "clear built files done ! & rebuilding ..."
fi

cd build && cmake -DCMAKE_BUILD_TYPE=MinSizeRel .. && make -j8

#!/bin/bash
cd build && rm -rf ./* \
         && echo "clear built files done ! & rebuilding ..." \
         && cmake .. && make -j8
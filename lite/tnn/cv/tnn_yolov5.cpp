//
// Created by DefTruth on 2021/11/6.
//

#include "tnn_yolov5.h"
#include "lite/utils.h"

using tnncv::TNNYoloV5;

TNNYoloV5::TNNYoloV5(const std::string &_proto_path,
                     const std::string &_model_path,
                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}


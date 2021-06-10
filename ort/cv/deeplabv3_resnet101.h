//
// Created by DefTruth on 2021/6/7.
//

#ifndef LITEHUB_ORT_CV_DEEPLABV3_RESNET101_H
#define LITEHUB_ORT_CV_DEEPLABV3_RESNET101_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class DeepLabV3ResNet101
  {
  private:
    ort::Env ort_env;
    ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> dynamic_input_node_dims; // >=1 inputs.
    unsigned int dynamic_input_height = 512;
    unsigned int dynamic_input_width = 512;
    unsigned int dynamic_input_tensor_size = 1;
    ort::MemoryInfo memory_info_handler = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    const char *onnx_path = nullptr;
    unsigned int num_outputs = 1;
    unsigned int num_inputs = 1;
    std::vector<float> dynamic_input_values_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // single input with dynamic height and width.
    explicit DeepLabV3ResNet101(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~DeepLabV3ResNet101();

  private:
    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    const float scale_vals[3] = {1.f / 0.229f, 1.f / 0.224f, 1.f / 0.225f};
    const char *class_names[20] = {
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    }; // 20 classes

  private:
    ort::Value transform(const cv::Mat &mat);

    void print_debug_string();

  public:
    void detect(const cv::Mat &mat, types::SegmentContent &content);

  };
}

#endif //LITEHUB_ORT_CV_DEEPLABV3_RESNET101_H

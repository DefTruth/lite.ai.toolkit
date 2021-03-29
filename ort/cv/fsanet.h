//
// Created by YanJun Qiu on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_FSANET_H
#define LITEHUB_ORT_CV_FSANET_H

#include "ort/core/ort_core.h"

namespace ortcv {

  class FSANet {
  private:
    ort::Env ort_env;
    ort::Session *ort_var_session = nullptr;
    ort::Session *ort_conv_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<int64_t> input_node_dims;
    std::size_t input_tensor_size = 1;
    std::vector<float> input_values_handler;
    ort::MemoryInfo memory_info_handler = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);  // mermory context
    std::vector<const char *> output_node_names = {"output"};
    const char *var_onnx_path = nullptr;
    const char *conv_onnx_path = nullptr;

  private:
    const unsigned int num_threads; // initialize at runtime
    // c++11 support const/static constexpr initialize inner class.
    static constexpr const float pad = 0.3f;
    static constexpr const int input_width = 64;
    static constexpr const int input_height = 64;

  public:
    FSANet(const std::string &_var_onnx_path, const std::string &_conv_onnx_path,
           unsigned int _num_threads = 1);

    ~FSANet();

    // un-copyable
  protected:
    FSANet(const FSANet &) = delete;

    FSANet(const FSANet &&) = delete;

    FSANet &operator=(const FSANet &) = delete;

    FSANet &operator=(const FSANet &&) = delete;

  private:
    ort::Value transform(const cv::Mat &mat); //  padding & resize & normalize.

  public:

    /**
     * detect the euler angles from a given face.
     * @param roi cv::Mat contains a single face.
     * @param euler_angles output (yaw,pitch,row).
     */
    void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
  };
}

#endif //LITEHUB_ORT_CV_FSANET_H

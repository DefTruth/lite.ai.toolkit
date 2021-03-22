//
// Created by YanJun Qiu on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_FSANET_H
#define LITEHUB_ORT_CV_FSANET_H

#include "ort/core/ort_core.h"

namespace ortcv {

  class FSANet {
    // onnxruntime模型加载相关
  private:
    ort::Env ort_env;
    ort::Session *ort_var_session = nullptr;
    ort::Session *ort_conv_session = nullptr;
    // 0. 输入节点的名称
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    // 1. 输入节点的维度
    std::vector<std::int64_t> input_node_dims;
    // 2. 输入数据的size
    std::size_t input_tensor_size = 0;
    // 3. 模型输入的数据
    std::vector<float> input_tensor_values;
    // 4. mermory info.
    ort::MemoryInfo memory_info = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // 5. 模型输出节点
    std::vector<const char *> output_node_names = {"output"};
    // 6. 模型路径
    const char *var_onnx_path;
    const char *conv_onnx_path;

    // 其他辅助参数
  private:
    // c++11 支持const成员类内初始化
    const float pad = 0.3f; // 0.3f
    const int input_width = 64;  // 64
    const int input_height = 64; // 64
    const bool use_padding = true; // true

    /**
     * padding & resize & normalize.
     */
    void preprocess(const cv::Mat &roi);

  public:
    explicit FSANet(const std::string &_var_onnx_path, const std::string &_conv_onnx_path);

    ~FSANet();

    // 7. 被禁止的构造函数
    FSANet(const FSANet &) = delete;

    FSANet(const FSANet &&) = delete;

    FSANet &operator=(const FSANet &) = delete;

    FSANet &operator=(const FSANet &&) = delete;

  public:

    /**
     * detect the euler angles from a given face.
     * @param roi cv::Mat contains a single face.
     * @param euler_angles output (yaw,pitch,row).
     */
    void detect(const cv::Mat &roi, std::vector<float> &euler_angles);

  };
}

#endif //LITEHUB_ORT_CV_FSANET_H

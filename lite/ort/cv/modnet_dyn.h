//
// Created by DefTruth on 2022/3/28.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_MODNET_DYNAMIC_H
#define LITE_AI_TOOLKIT_ORT_CV_MODNET_DYNAMIC_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS MODNetDyn
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> dynamic_input_node_dims; // >=1 inputs.
    unsigned int dynamic_input_height = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_width = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_tensor_size = 1; // init only, will change according to input mat.
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    unsigned int num_outputs = 1;
    unsigned int num_inputs = 1;
    std::vector<float> dynamic_input_values_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // single input with dynamic height and width.
    explicit MODNetDyn(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~MODNetDyn();

  protected:
    MODNetDyn(const MODNetDyn &) = delete;

    MODNetDyn(MODNetDyn &&) = delete;

    MODNetDyn &operator=(const MODNetDyn &) = delete;

    MODNetDyn &operator=(MODNetDyn &&) = delete;

  private:
    static constexpr const float mean_val = 127.5f; // RGB
    static constexpr const float scale_val = 1.f / 127.5f;
    static constexpr const unsigned int align_val = 32;

  private:
    Ort::Value transform(const cv::Mat &mat);

    void print_debug_string();

    void update_dynamic_shape(unsigned int img_height, unsigned int img_width);

    cv::Mat padding(const cv::Mat &unpad_mat);

    void generate_matting(std::vector<Ort::Value> &output_tensors,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false);

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_MODNET_DYNAMIC_H

//
// Created by DefTruth on 2022/6/29.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_FACE_PARSING_BISENET_DYN_H
#define LITE_AI_TOOLKIT_ORT_CV_FACE_PARSING_BISENET_DYN_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS FaceParsingBiSeNetDyn
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char *> input_node_names;
    std::vector<std::string> input_node_names_;
    std::vector<std::vector<int64_t>> dynamic_input_node_dims; // >=1 inputs.
    unsigned int dynamic_input_height = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_width = 512; // init only, will change according to input mat.
    unsigned int dynamic_input_tensor_size = 1; // init only, will change according to input mat.
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::string> output_node_names_;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    unsigned int num_outputs = 3;
    unsigned int num_inputs = 1;
    std::vector<float> dynamic_input_values_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    // single input with dynamic height and width.
    explicit FaceParsingBiSeNetDyn(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~FaceParsingBiSeNetDyn();

  protected:
    FaceParsingBiSeNetDyn(const FaceParsingBiSeNetDyn &) = delete;

    FaceParsingBiSeNetDyn(FaceParsingBiSeNetDyn &&) = delete;

    FaceParsingBiSeNetDyn &operator=(const FaceParsingBiSeNetDyn &) = delete;

    FaceParsingBiSeNetDyn &operator=(FaceParsingBiSeNetDyn &&) = delete;

  private:
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; // RGB
    const float scale_vals[3] = {1.f / (0.229f * 255.f), 1.f / (0.224f * 255.f), 1.f / (0.225f * 255.f)};
    static constexpr const unsigned int align_val = 32;

  private:
    Ort::Value transform(const cv::Mat &mat);

    void print_debug_string();

    void update_dynamic_shape(unsigned int img_height, unsigned int img_width);

    cv::Mat padding(const cv::Mat &unpad_mat);

    void generate_mask(std::vector<Ort::Value> &output_tensors,
                       const cv::Mat &mat, types::FaceParsingContent &content,
                       bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::FaceParsingContent &content,
                bool minimum_post_process = false);
  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_FACE_PARSING_BISENET_DYN_H

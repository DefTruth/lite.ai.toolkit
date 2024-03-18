//
// Created by DefTruth on 2022/6/3.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_HEAD_SEG_H
#define LITE_AI_TOOLKIT_ORT_CV_HEAD_SEG_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS HeadSeg
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::string> input_node_names_;
    std::vector<int64_t> input_node_dims; // 1 input only. (?,384,384,3)
    std::size_t input_tensor_size = 1;
    std::vector<float> input_values_handler;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::string> output_node_names_;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    int num_outputs = 1;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    explicit HeadSeg(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~HeadSeg();

  protected:
    HeadSeg(const HeadSeg &) = delete;

    HeadSeg(HeadSeg &&) = delete;

    HeadSeg &operator=(const HeadSeg &) = delete;

    HeadSeg &operator=(HeadSeg &&) = delete;

  private:
    void initialize_handler();

    void print_debug_string();

  private:
    Ort::Value transform(const cv::Mat &mat_rs);

  public:
    void detect(const cv::Mat &mat, types::HeadSegContent &content);

  };
}
#endif //LITE_AI_TOOLKIT_ORT_CV_HEAD_SEG_H

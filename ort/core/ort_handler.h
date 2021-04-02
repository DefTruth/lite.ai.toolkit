//
// Created by DefTruth on 2021/3/30.
//

#ifndef LITEHUB_ORT_CORE_ORT_HANDLER_H
#define LITEHUB_ORT_CORE_ORT_HANDLER_H

#include "__ort_core.h"
#include "ort_types.h"

// global
namespace core {
  // single input & multi outputs.
  class BasicOrtHandler {
  protected:
    ort::Env ort_env;
    ort::Session *ort_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<int64_t> input_node_dims; // 1 input only.
    std::size_t input_tensor_size = 1;
    std::vector<float> input_values_handler;
    ort::MemoryInfo memory_info_handler = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    const char *onnx_path = nullptr;
    int num_outputs = 1;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  protected:
    BasicOrtHandler(const std::string &_onnx_path, unsigned int _num_threads = 1);

    virtual ~BasicOrtHandler();

    // un-copyable
  protected:
    BasicOrtHandler(const BasicOrtHandler &) = delete;

    BasicOrtHandler(BasicOrtHandler &&) = delete;

    BasicOrtHandler &operator=(const BasicOrtHandler &) = delete;

    BasicOrtHandler &operator=(BasicOrtHandler &&) = delete;

  protected:
    virtual ort::Value transform(const cv::Mat &mat) = 0;

    // TODO: implemetation for BasicOrtHandler::detect_().

  private:
    void initialize_handler();

    void print_debug_string();
  };

  // multi inputs & multi outputs.
  class BasicMultiOrtHandler {
  protected:
    ort::Env ort_env;
    ort::Session *ort_session = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes;
    std::vector<std::vector<float>> input_values_handlers; // multi handlers.
    ort::MemoryInfo memory_info_handler = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    const char *onnx_path = nullptr;
    int num_outputs = 1;
    int num_inputs = 1;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  protected:
    BasicMultiOrtHandler(const std::string &_onnx_path, unsigned int _num_threads = 1);

    virtual ~BasicMultiOrtHandler();

    // un-copyable
  protected:
    BasicMultiOrtHandler(const BasicMultiOrtHandler &) = delete;

    BasicMultiOrtHandler(BasicMultiOrtHandler &&) = delete;

    BasicMultiOrtHandler &operator=(const BasicMultiOrtHandler &) = delete;

    BasicMultiOrtHandler &operator=(BasicMultiOrtHandler &&) = delete;

  protected:
    virtual std::vector<ort::Value> transform(const std::vector<cv::Mat> &mats) = 0;

    // TODO: implemetation for BasicMultiOrtHandler::detect_().

  private:
    void initialize_handler();

    void print_debug_string();
  };
}

#endif //LITEHUB_ORT_CORE_ORT_HANDLER_H

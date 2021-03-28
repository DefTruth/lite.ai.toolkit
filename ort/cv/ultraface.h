//
// Created by YanJun Qiu on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_ULTRAFACE_H
#define LITEHUB_ORT_CV_ULTRAFACE_H

#include "ort/core/ort_core.h"

namespace ortcv {

  class UltraFace {
  private:
    ort::Env ort_env;
    ort::Session *ort_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<int64_t> input_node_dims; // 1 input only.
    std::size_t input_tensor_size = 1;
    std::vector<float> input_tensor_values;
    ort::MemoryInfo memory_info = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // 2 outputs
    const char *onnx_path = nullptr;
    int num_outputs;

  private:
    const int input_width; // initialize at runtime. 320 | 640
    const int input_height;  // initialize at runtime. 240 | 480
    const unsigned int num_threads; // initialize at runtime.

    static constexpr const float mean_val = 127.0f;
    static constexpr const float scale_val = 1.0 / 128.0f;
    enum NMS {HARD=0, BLEND=1};

  public:
    UltraFace(const std::string &_onnx_path, int _input_height, int _input_width,
              unsigned int _num_threads = 1);

    ~UltraFace();

    // un-copyable
  protected:
    UltraFace(const UltraFace &) = delete;

    UltraFace(const UltraFace &&) = delete;

    UltraFace &operator=(const UltraFace &) = delete;

    UltraFace &operator=(const UltraFace &&) = delete;

  private:
    void preprocess(const cv::Mat &mat); // padding & resize & normalize.

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.7f, float iou_threshold = 0.3f,
                unsigned int topk = 100, unsigned int nms_type = 0);
  };
}

#endif //LITEHUB_ORT_CV_ULTRAFACE_H

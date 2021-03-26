//
// Created by YanJun Qiu on 2021/3/14.
//

#ifndef LITEHUB_ORT_CV_ULTRAFACE_H
#define LITEHUB_ORT_CV_ULTRAFACE_H

#include "ort/core/ort_core.h"

namespace ortcv {

  typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
  } Box;

  /**
   * reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
   */

  class UltraFace {
  private:
    ort::Env ort_env;
    ort::Session *ort_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<std::int64_t> input_node_dims; // 1 input only.
    std::size_t input_tensor_size = 1;
    std::vector<float> input_tensor_values;
    ort::MemoryInfo memory_info = ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<std::int64_t>> output_node_dims; // 2 outputs
    const char *onnx_path = nullptr;
    int num_outputs;

  private:
    const int input_width; // init on runtime.
    const int input_height;  // init on runtime.
    const unsigned int num_threads; // init on runtime.

    static constexpr const float mean_val = 127.0f;
    static constexpr const float scale_val = 1.0 / 128.0f;

  public:
    UltraFace(const std::string &_onnx_path, int _input_height, int _input_width,
              unsigned int _num_threads = 1);

    ~UltraFace();

  protected:
    UltraFace(const UltraFace &) = delete;

    UltraFace(const UltraFace &&) = delete;

    UltraFace &operator=(const UltraFace &) = delete;

    UltraFace &operator=(const UltraFace &&) = delete;

  private:
    /**
     * padding & resize & normalize.
     */
    void preprocess(const cv::Mat &mat);

    void nms(std::vector<Box> &input, std::vector<Box> &output, __unused int type);

    void generate_bounding_boxes(std::vector<Box> &bbox_collection,
                                 const float *scores,
                                 const float *boxes,
                                 float score_threshold, int num_anchors);

  public:
    void detect(const cv::Mat &mat, std::vector<Box> &detected_boxes,
                float score_threshold = 0.7f, float iou_threshold = 0.3f,
                int top_k = 100);

    static void draw_boxes_inplane(cv::Mat &mat_inplane, const std::vector<Box> &_boxes);

    static cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<Box> &_boxes);
  };
}

#endif //LITEHUB_ORT_CV_ULTRAFACE_H

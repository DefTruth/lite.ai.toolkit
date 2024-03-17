//
// Created by DefTruth on 2022/4/9.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_BACKGROUNDMATTINGV2_H
#define LITE_AI_TOOLKIT_ORT_CV_BACKGROUNDMATTINGV2_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS BackgroundMattingV2
  {
  private:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    // CPU MemoryInfo
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    // hardcode input node names
    unsigned int num_inputs = 2;
    std::vector<const char *> input_node_names;
    std::vector<std::string> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims;
    // hardcode output node names
    unsigned int num_outputs = 6;
    std::vector<const char *> output_node_names;
    std::vector<std::string> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims;
    const LITEORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    // input values handlers
    std::vector<float> input_mat_value_handler;
    std::vector<float> input_bgr_value_handler;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  public:
    explicit BackgroundMattingV2(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~BackgroundMattingV2();

  protected:
    BackgroundMattingV2(const BackgroundMattingV2 &) = delete; //
    BackgroundMattingV2(BackgroundMattingV2 &&) = delete; //
    BackgroundMattingV2 &operator=(const BackgroundMattingV2 &) = delete; //
    BackgroundMattingV2 &operator=(BackgroundMattingV2 &&) = delete; //

  private:
    static constexpr const float mean_val = 0.f; // RGB
    static constexpr const float scale_val = 1.f / 255.f;

  private:
    std::vector<Ort::Value> transform(const cv::Mat &mat, const cv::Mat &bgr);

    void generate_matting(std::vector<Ort::Value> &output_tensors,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false, bool minimum_post_process = false);

    void print_debug_string();

  public:
    /**
     * @param mat cv::Mat input image with BGR format.
     * @param bgr cv::Mat input background image with BGR format.
     * @param content MattingContent output fgr, pha and merge_mat (if minimum_post_process is false)
     * @param remove_noise bool, whether to remove small connected areas.
     * @param minimum_post_process bool, will not return demo merge mat if True.
     */
    void detect(const cv::Mat &mat, const cv::Mat &bgr, types::MattingContent &content,
                bool remove_noise = false, bool minimum_post_process = false);

  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_BACKGROUNDMATTINGV2_H

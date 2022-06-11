//
// Created by DefTruth on 2022/6/11.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_HEAD_SEG_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_HEAD_SEG_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNHeadSeg
  {
  public:
    explicit MNNHeadSeg(const std::string &_mnn_path, unsigned int _num_threads = 1);

    ~MNNHeadSeg();

  private:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // assume single input.
    MNN::ScheduleConfig schedule_config;
    const char *mnn_path = nullptr;
    const char *log_id = nullptr;
    const unsigned int num_threads; // initialize at runtime.
    int dimension_type; // hint only

  private:
    // hardcode input size
    static constexpr const int input_batch = 1;
    static constexpr const int input_channel = 3;
    static constexpr const int input_height = 384;
    static constexpr const int input_width = 384;

  private:
    void transform(const cv::Mat &mat_rs);

    void print_debug_string();

  public:
    void detect(const cv::Mat &mat, types::HeadSegContent &content);
  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_HEAD_SEG_H

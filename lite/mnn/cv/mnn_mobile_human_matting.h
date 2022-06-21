//
// Created by DefTruth on 2022/6/20.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HUMAN_MATTING_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HUMAN_MATTING_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMobileHumanMatting : public BasicMNNHandler
  {
  public:
    explicit MNNMobileHumanMatting(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMobileHumanMatting() override = default;

  private:
    const float mean_vals[3] = {104.f, 112.f, 121.f}; //BGR
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat) override; // resize & normalize.

    void generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false, bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false,
                bool minimum_post_process = false);

  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MOBILE_HUMAN_MATTING_H

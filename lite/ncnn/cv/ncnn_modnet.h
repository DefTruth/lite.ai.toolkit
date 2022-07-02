//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_NCNN_CV_NCNN_MODNET_H
#define LITE_AI_TOOLKIT_NCNN_CV_NCNN_MODNET_H

#include "lite/ncnn/core/ncnn_core.h"

namespace ncnncv
{
  class LITE_EXPORTS NCNNMODNet : public BasicNCNNHandler
  {
  public:
    explicit NCNNMODNet(const std::string &_param_path,
                        const std::string &_bin_path,
                        unsigned int _num_threads = 1,
                        unsigned int _input_height = 512,
                        unsigned int _input_width = 512);

    ~NCNNMODNet() override = default;

  private:
    const int input_height;
    const int input_width;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

    void generate_matting(ncnn::Extractor &extractor,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false, bool minimum_post_process = false);

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false,
                bool minimum_post_process = false);

  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_MODNET_H

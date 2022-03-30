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
                        unsigned int _input_height = 512,
                        unsigned int _input_width = 512,
                        unsigned int _num_threads = 1);

    ~NCNNMODNet() override = default;

  private:
    const int input_height;
    const int input_width;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void transform(const cv::Mat &mat, ncnn::Mat &in) override;

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(ncnn::Extractor &extractor,
                          const cv::Mat &mat, types::MattingContent &content,
                          bool remove_noise = false);

  public:
    void detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise = false);

  public:
    // class method.
    static void swap_background(const cv::Mat &fg_mat, const cv::Mat &pha_mat,
                                const cv::Mat &bg_mat, cv::Mat &out_mat);
  };
}

#endif //LITE_AI_TOOLKIT_NCNN_CV_NCNN_MODNET_H

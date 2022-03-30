//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_MODNET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_MODNET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNMODNet : public BasicMNNHandler
  {
  public:
    explicit MNNMODNet(const std::string &_mnn_path, unsigned int _num_threads = 1); //
    ~MNNMODNet() override = default;

  private:
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};

  private:
    void initialize_pretreat();

    void transform(const cv::Mat &mat) override; // resize & normalize.

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(const std::map<std::string, MNN::Tensor *> &output_tensors,
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

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_MODNET_H

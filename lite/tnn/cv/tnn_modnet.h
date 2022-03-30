//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_MODNET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_MODNET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNMODNet : public BasicTNNHandler
  {
  public:
    explicit TNNMODNet(const std::string &_proto_path,
                       const std::string &_model_path,
                       unsigned int _num_threads = 1);

    ~TNNMODNet() override = default;

  private:
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    std::vector<float> bias_vals = {-1.f, -1.f, -1.f};

  private:
    void transform(const cv::Mat &mat_rs) override; //

    void remove_small_connected_area(cv::Mat &alpha_pred);

    void generate_matting(std::shared_ptr<tnn::Instance> &_instance,
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

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_MODNET_H

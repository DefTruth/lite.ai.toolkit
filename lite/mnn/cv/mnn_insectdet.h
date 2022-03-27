//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_INSECTDET_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_INSECTDET_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNInsectDet : public BasicMNNHandler
  {
  public:
    explicit MNNInsectDet(const std::string &_mnn_path, unsigned int _num_threads = 1);

    ~MNNInsectDet() override = default;

  private:
    // nested classes
    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } InsectDetScaleParams;

  private:
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat_rs) override; // without resize

    void initialize_pretreat();

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        InsectDetScaleParams &scale_params);

    void generate_bboxes(const InsectDetScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         const std::map<std::string, MNN::Tensor *> &output_tensors,
                         float score_threshold, int img_height,
                         int img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.5f, float iou_threshold = 0.45f,
                unsigned int topk = 100);

  };
}

#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_INSECTDET_H

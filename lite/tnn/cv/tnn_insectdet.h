//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_TNN_CV_TNN_INSECTDET_H
#define LITE_AI_TOOLKIT_TNN_CV_TNN_INSECTDET_H

#include "lite/tnn/core/tnn_core.h"

namespace tnncv
{
  class LITE_EXPORTS TNNInsectDet : public BasicTNNHandler
  {
  public:
    explicit TNNInsectDet(const std::string &_proto_path,
                          const std::string &_model_path,
                          unsigned int _num_threads = 1);

    ~TNNInsectDet() override = default;

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
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    std::vector<float> bias_vals = {0.f, 0.f, 0.f}; // RGB
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        InsectDetScaleParams &scale_params);

    void generate_bboxes(const InsectDetScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::shared_ptr<tnn::Instance> &_instance,
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

#endif //LITE_AI_TOOLKIT_TNN_CV_TNN_INSECTDET_H

//
// Created by DefTruth on 2022/3/27.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_INSECTDET_H
#define LITE_AI_TOOLKIT_ORT_CV_INSECTDET_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  class LITE_EXPORTS InsectDet : public BasicOrtHandler
  {
  public:
    explicit InsectDet(const std::string &_onnx_path, unsigned int _num_threads = 1);

    ~InsectDet() override = default;

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
    static constexpr const float mean_val = 0.f; // RGB
    static constexpr const float scale_val = 1.0 / 255.f;
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        InsectDetScaleParams &scale_params);

    void generate_bboxes(const InsectDetScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
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

#endif //LITE_AI_TOOLKIT_ORT_CV_INSECTDET_H

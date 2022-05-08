//
// Created by DefTruth on 2022/5/8.
//

#ifndef LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_BLAZEFACE_H
#define LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_BLAZEFACE_H

#include "lite/mnn/core/mnn_core.h"

namespace mnncv
{
  class LITE_EXPORTS MNNYOLOv5BlazeFace : public BasicMNNHandler
  {
  public:
    explicit MNNYOLOv5BlazeFace(const std::string &_mnn_path, unsigned int _num_threads = 1);

    ~MNNYOLOv5BlazeFace() override = default;

  private:
    // nested classes
    typedef struct
    {
      float ratio;
      int dw;
      int dh;
      bool flag;
    } YOLOv5BlazeFaceScaleParams;

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
                        YOLOv5BlazeFaceScaleParams &scale_params);

    void generate_bboxes_kps(const YOLOv5BlazeFaceScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             const std::map<std::string, MNN::Tensor *> &output_tensors,
                             float score_threshold, float img_height,
                             float img_width); // rescale & exclude

    void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                        std::vector<types::BoxfWithLandmarks> &output,
                        float iou_threshold, unsigned int topk);

  public:
    void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.3f, float iou_threshold = 0.45f,
                unsigned int topk = 400);

  };
}


#endif //LITE_AI_TOOLKIT_MNN_CV_MNN_YOLOV5_BLAZEFACE_H

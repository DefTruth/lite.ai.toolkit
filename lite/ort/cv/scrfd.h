//
// Created by DefTruth on 2021/12/30.
//

#ifndef LITE_AI_TOOLKIT_ORT_CV_SCRFD_H
#define LITE_AI_TOOLKIT_ORT_CV_SCRFD_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
  // https://github.com/deepinsight/insightface/blob/master/detection/scrfd/
  // mmdet/core/anchor/anchor_generator.py
  class LITE_EXPORTS SCRFD : public BasicOrtHandler
  {
  public:
    explicit SCRFD(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicOrtHandler(_onnx_path, _num_threads)
    {};

    ~SCRFD() override = default;

  private:
    // nested classes
    struct SCRFDPoint
    {
      float cx;
      float cy;
      float stride;
    };

  private:
    // blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float scale_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
    unsigned int fmc = 3; // feature map count
    bool use_kps = false;
    unsigned int num_anchors = 2;
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
    // if num_anchors>1, then stack points in col major -> (height,num_anchor*width,2)
    // anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;

    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    static constexpr const unsigned int max_nms = 30000;

  private:
    Ort::Value transform(const cv::Mat &mat) override; //

    // initial steps and num_anchors
    // https://github.com/deepinsight/insightface/blob/master/detection/scrfd/tools/scrfd.py
    void initial_context();

    // generate once.
    void generate_points(const int target_height, const int target_width);

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<Ort::Value> &output_tensors,
                         float score_threshold, float img_height,
                         float img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 400, unsigned int nms_type = NMS::HARD);

  };
}

#endif //LITE_AI_TOOLKIT_ORT_CV_SCRFD_H

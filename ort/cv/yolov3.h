//
// Created by DefTruth on 2021/5/23.
//

#ifndef LITEHUB_ORT_CV_YOLOV3_H
#define LITEHUB_ORT_CV_YOLOV3_H

#include "ort/core/ort_core.h"

namespace ortcv
{
  class YoloV3 : public BasicMultiOrtHandler
  {
  public:
    explicit YoloV3(const std::string &_onnx_path, unsigned int _num_threads = 1) :
        BasicMultiOrtHandler(_onnx_path, _num_threads)
    {};

    ~YoloV3() override = default;

  private:
    static constexpr const float mean_val = 0.f;
    static constexpr const float scale_val = 1.0 / 255.f;
    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };

  private:
    std::vector<ort::Value> transform(const std::vector<cv::Mat> &mats) override;

    void generate_bboxes(std::vector<types::Boxf> &bbox_collection,
                         std::vector<ort::Value> &output_tensors,
                         float img_height, float img_width); // rescale & exclude

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);

  };
}

#endif //LITEHUB_ORT_CV_YOLOV3_H

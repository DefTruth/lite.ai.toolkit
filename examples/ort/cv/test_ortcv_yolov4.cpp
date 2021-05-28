//
// Created by DefTruth on 2021/5/28.
//

#include <iostream>
#include <vector>

#include "ort/cv/yolov4.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_yolov4()
{

  std::string onnx_path = "../../../hub/onnx/cv/voc-mobilenetv2-yolov4-640.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_yolov4.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_yolov4.jpg";

  ortcv::YoloV4 *yolov4 = new ortcv::YoloV4(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov4->detect(img_bgr, detected_boxes);

  ortcv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov4;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_yolov4();
  return 0;
}

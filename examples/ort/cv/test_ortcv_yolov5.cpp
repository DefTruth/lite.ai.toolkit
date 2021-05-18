//
// Created by yanjun qiu on 2021/5/18.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedLocalVariable"
#include <iostream>
#include <vector>

#include "ort/cv/yolov5.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_yolov5()
{

  std::string onnx_path = "../../../hub/onnx/cv/yolov5s.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_yolov5.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_yolov5.jpg";

  ortcv::YoloV5 *yolov5 = new ortcv::YoloV5(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov5->detect(img_bgr, detected_boxes);

  ortcv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete yolov5;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_yolov5();
  return 0;
}

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
  std::string test_img_path_1 = "../../../examples/ort/resources/test_ortcv_yolov5_1.jpg";
  std::string test_img_path_2 = "../../../examples/ort/resources/test_ortcv_yolov5_2.jpg";
  std::string save_img_path_1 = "../../../logs/test_ortcv_yolov5_1.jpg";
  std::string save_img_path_2 = "../../../logs/test_ortcv_yolov5_2.jpg";

  ortcv::YoloV5 *yolov5 = new ortcv::YoloV5(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes_1;
  cv::Mat img_bgr_1 = cv::imread(test_img_path_1);
  yolov5->detect(img_bgr_1, detected_boxes_1);

  ortcv::utils::draw_boxes_inplace(img_bgr_1, detected_boxes_1);

  cv::imwrite(save_img_path_1, img_bgr_1);

  std::cout << "Detected Boxes Num: " << detected_boxes_1.size() << std::endl;

  std::vector<ortcv::types::Boxf> detected_boxes_2;
  cv::Mat img_bgr_2 = cv::imread(test_img_path_2);
  yolov5->detect(img_bgr_2, detected_boxes_2);

  ortcv::utils::draw_boxes_inplace(img_bgr_2, detected_boxes_2);

  cv::imwrite(save_img_path_2, img_bgr_2);

  std::cout << "Detected Boxes Num: " << detected_boxes_2.size() << std::endl;

  delete yolov5;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_yolov5();
  return 0;
}

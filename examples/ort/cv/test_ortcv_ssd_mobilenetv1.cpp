//
// Created by DefTruth on 2021/6/6.
//

#include <iostream>
#include <vector>

#include "ort/cv/ssd_mobilenetv1.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_ssd_mobilenetv1()
{

  std::string onnx_path = "../../../hub/onnx/cv/ssd_mobilenet_v1_10.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_ssd_mobilenetv1.png";
  std::string save_img_path = "../../../logs/test_ortcv_ssd_mobilenetv1.jpg";

  ortcv::SSDMobileNetV1 *ssd_mobilenetv1 = new ortcv::SSDMobileNetV1(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssd_mobilenetv1->detect(img_bgr, detected_boxes);

  ortcv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete ssd_mobilenetv1;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_ssd_mobilenetv1();
  return 0;
}
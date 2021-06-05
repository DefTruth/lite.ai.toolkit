//
// Created by DefTruth on 2021/6/5.
//

#include <iostream>
#include <vector>

#include "ort/cv/ssd.h"
#include "ort/core/ort_utils.h"


static void test_ortcv_ssd()
{

  std::string onnx_path = "../../../hub/onnx/cv/ssd-10.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_ssd.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_ssd.jpg";

  ortcv::SSD *ssd = new ortcv::SSD(onnx_path);

  std::vector<ortcv::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssd->detect(img_bgr, detected_boxes);

  ortcv::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;

  delete ssd;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_ssd();
  return 0;
}

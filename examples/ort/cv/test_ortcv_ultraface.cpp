//
// Created by YanJun Qiu on 2021/3/27.
//

#include <iostream>
#include <vector>

#include "ort/cv/ultraface.h"


static void test_ortcv_ultraface() {

  std::string onnx_path = "../../../hub/onnx/cv/ultraface-rfb-640.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_ultraface.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_ultraface.jpg";

  ortcv::UltraFace *ultraface = new ortcv::UltraFace(onnx_path, 480, 640, 1);

  std::vector<ortcv::UltraBox> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ultraface->detect(img_bgr, detected_boxes);

  ortcv::UltraFace::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Face Num: " << detected_boxes.size() << std::endl;

  delete ultraface;

}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_ultraface();
  return 0;
}
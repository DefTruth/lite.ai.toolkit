//
// Created by DefTruth on 2021/4/4.
//

#include <iostream>
#include <vector>

#include "ort/cv/arcafce_resnet.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_arcface_resnet() {
  std::string onnx_path = "../../../hub/onnx/cv/arcfaceresnet100-8.onnx";
  std::string test_img_path0 = "../../../examples/ort/resources/test_ortcv_arcface_resnet_0.png";
  std::string test_img_path1 = "../../../examples/ort/resources/test_ortcv_arcface_resnet_1.png";

  ortcv::ArcFaceResNet *arcface_resnet = new ortcv::ArcFaceResNet(onnx_path);

  ortcv::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  arcface_resnet->detect(img_bgr0, face_content0);
  arcface_resnet->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag) {
    float sim = ortcv::utils::math::cosine_similarity<float>(face_content0.embedding, face_content1.embedding);
    std::cout << "Detected Sim: " << sim << std::endl;
  }

  delete arcface_resnet;
}

int main(__unused int argc, __unused char *argv[]) {
  test_ortcv_arcface_resnet();
  return 0;
  //  Ort::Exception: Non-zero status code returned while running BatchNormalization node. Name:'stage1_unit1_bn1' Status Message: Invalid input scale: NumDimensions() != 3
}
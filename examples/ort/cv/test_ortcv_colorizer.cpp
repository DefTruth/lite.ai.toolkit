//
// Created by DefTruth on 2021/4/9.
//

#include <iostream>
#include <vector>

#include "ort/cv/colorizer.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_colorizer()
{
  std::string eccv16_onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string siggraph17_onnx_path = "../../../hub/onnx/cv/siggraph17-colorizer.onnx";
  std::string test_img_path1 = "../../../examples/ort/resources/test_ortcv_colorizer_1.jpg";
  std::string test_img_path2 = "../../../examples/ort/resources/test_ortcv_colorizer_2.jpg";
  std::string test_img_path3 = "../../../examples/ort/resources/test_ortcv_colorizer_3.jpg";
  std::string save_eccv_img_path1 = "../../../logs/test_ortcv_eccv16_colorizer_1.jpg";
  std::string save_eccv_img_path2 = "../../../logs/test_ortcv_eccv16_colorizer_2.jpg";
  std::string save_eccv_img_path3 = "../../../logs/test_ortcv_eccv16_colorizer_3.jpg";
  std::string save_siggraph_img_path1 = "../../../logs/test_ortcv_siggraph17_colorizer_1.jpg";
  std::string save_siggraph_img_path2 = "../../../logs/test_ortcv_siggraph17_colorizer_2.jpg";
  std::string save_siggraph_img_path3 = "../../../logs/test_ortcv_siggraph17_colorizer_3.jpg";

  ortcv::Colorizer *eccv16_colorizer = new ortcv::Colorizer(eccv16_onnx_path);
  ortcv::Colorizer *siggraph17_colorizer = new ortcv::Colorizer(siggraph17_onnx_path);

  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  cv::Mat img_bgr3 = cv::imread(test_img_path3);
  ortcv::types::ColorizeContent eccv16_colorize_content1;
  ortcv::types::ColorizeContent eccv16_colorize_content2;
  ortcv::types::ColorizeContent eccv16_colorize_content3;
  ortcv::types::ColorizeContent siggraph17_colorize_content1;
  ortcv::types::ColorizeContent siggraph17_colorize_content2;
  ortcv::types::ColorizeContent siggraph17_colorize_content3;
  eccv16_colorizer->detect(img_bgr1, eccv16_colorize_content1);
  eccv16_colorizer->detect(img_bgr2, eccv16_colorize_content2);
  eccv16_colorizer->detect(img_bgr3, eccv16_colorize_content3);
  siggraph17_colorizer->detect(img_bgr1, siggraph17_colorize_content1);
  siggraph17_colorizer->detect(img_bgr2, siggraph17_colorize_content2);
  siggraph17_colorizer->detect(img_bgr3, siggraph17_colorize_content3);

  if (eccv16_colorize_content1.flag) cv::imwrite(save_eccv_img_path1, eccv16_colorize_content1.mat);
  if (eccv16_colorize_content2.flag) cv::imwrite(save_eccv_img_path2, eccv16_colorize_content2.mat);
  if (eccv16_colorize_content3.flag) cv::imwrite(save_eccv_img_path3, eccv16_colorize_content3.mat);
  if (siggraph17_colorize_content1.flag) cv::imwrite(save_siggraph_img_path1, siggraph17_colorize_content1.mat);
  if (siggraph17_colorize_content2.flag) cv::imwrite(save_siggraph_img_path2, siggraph17_colorize_content2.mat);
  if (siggraph17_colorize_content3.flag) cv::imwrite(save_siggraph_img_path3, siggraph17_colorize_content3.mat);

  std::cout << "Colorization Done." << std::endl;

  delete eccv16_colorizer;
  delete siggraph17_colorizer;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_colorizer();
  return 0;
}
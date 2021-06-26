//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string eccv16_onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string siggraph17_onnx_path = "../../../hub/onnx/cv/siggraph17-colorizer.onnx";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string test_img_path2 = "../../../examples/lite/resources/test_lite_colorizer_2.jpg";
  std::string test_img_path3 = "../../../examples/lite/resources/test_lite_colorizer_3.jpg";
  std::string test_img_path4 = "../../../examples/lite/resources/test_lite_colorizer_one_piece_0.png";
  std::string save_eccv_img_path1 = "../../../logs/test_lite_eccv16_colorizer_1.jpg";
  std::string save_eccv_img_path2 = "../../../logs/test_lite_eccv16_colorizer_2.jpg";
  std::string save_eccv_img_path3 = "../../../logs/test_lite_eccv16_colorizer_3.jpg";
  std::string save_eccv_img_path4 = "../../../logs/test_lite_eccv16_colorizer_one_piece_0.jpg";
  std::string save_siggraph_img_path1 = "../../../logs/test_lite_siggraph17_colorizer_1.jpg";
  std::string save_siggraph_img_path2 = "../../../logs/test_lite_siggraph17_colorizer_2.jpg";
  std::string save_siggraph_img_path3 = "../../../logs/test_lite_siggraph17_colorizer_3.jpg";
  std::string save_siggraph_img_path4 = "../../../logs/test_lite_siggraph17_colorizer_one_piece_0.jpg";

  lite::cv::colorization::Colorizer *eccv16_colorizer =
      new lite::cv::colorization::Colorizer(eccv16_onnx_path);
  lite::cv::colorization::Colorizer *siggraph17_colorizer =
      new lite::cv::colorization::Colorizer(siggraph17_onnx_path);

  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  cv::Mat img_bgr3 = cv::imread(test_img_path3);
  cv::Mat img_bgr4 = cv::imread(test_img_path4);

  lite::cv::types::ColorizeContent eccv16_colorize_content1;
  lite::cv::types::ColorizeContent eccv16_colorize_content2;
  lite::cv::types::ColorizeContent eccv16_colorize_content3;
  lite::cv::types::ColorizeContent eccv16_colorize_content4;

  lite::cv::types::ColorizeContent siggraph17_colorize_content1;
  lite::cv::types::ColorizeContent siggraph17_colorize_content2;
  lite::cv::types::ColorizeContent siggraph17_colorize_content3;
  lite::cv::types::ColorizeContent siggraph17_colorize_content4;

  eccv16_colorizer->detect(img_bgr1, eccv16_colorize_content1);
  eccv16_colorizer->detect(img_bgr2, eccv16_colorize_content2);
  eccv16_colorizer->detect(img_bgr3, eccv16_colorize_content3);
  eccv16_colorizer->detect(img_bgr4, eccv16_colorize_content4);

  siggraph17_colorizer->detect(img_bgr1, siggraph17_colorize_content1);
  siggraph17_colorizer->detect(img_bgr2, siggraph17_colorize_content2);
  siggraph17_colorizer->detect(img_bgr3, siggraph17_colorize_content3);
  siggraph17_colorizer->detect(img_bgr4, siggraph17_colorize_content4);

  if (eccv16_colorize_content1.flag) cv::imwrite(save_eccv_img_path1, eccv16_colorize_content1.mat);
  if (eccv16_colorize_content2.flag) cv::imwrite(save_eccv_img_path2, eccv16_colorize_content2.mat);
  if (eccv16_colorize_content3.flag) cv::imwrite(save_eccv_img_path3, eccv16_colorize_content3.mat);
  if (eccv16_colorize_content4.flag) cv::imwrite(save_eccv_img_path4, eccv16_colorize_content4.mat);

  if (siggraph17_colorize_content1.flag) cv::imwrite(save_siggraph_img_path1, siggraph17_colorize_content1.mat);
  if (siggraph17_colorize_content2.flag) cv::imwrite(save_siggraph_img_path2, siggraph17_colorize_content2.mat);
  if (siggraph17_colorize_content3.flag) cv::imwrite(save_siggraph_img_path3, siggraph17_colorize_content3.mat);
  if (siggraph17_colorize_content4.flag) cv::imwrite(save_siggraph_img_path4, siggraph17_colorize_content4.mat);

  std::cout << "Default Version Colorization Done." << std::endl;

  delete eccv16_colorizer;
  delete siggraph17_colorizer;
}

static void test_onnxruntime()
{
  std::string eccv16_onnx_path = "../../../hub/onnx/cv/eccv16-colorizer.onnx";
  std::string siggraph17_onnx_path = "../../../hub/onnx/cv/siggraph17-colorizer.onnx";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_colorizer_1.jpg";
  std::string test_img_path2 = "../../../examples/lite/resources/test_lite_colorizer_2.jpg";
  std::string test_img_path3 = "../../../examples/lite/resources/test_lite_colorizer_3.jpg";
  std::string test_img_path4 = "../../../examples/lite/resources/test_lite_colorizer_one_piece_0.png";
  std::string save_eccv_img_path1 = "../../../logs/test_onnx_eccv16_colorizer_1.jpg";
  std::string save_eccv_img_path2 = "../../../logs/test_onnx_eccv16_colorizer_2.jpg";
  std::string save_eccv_img_path3 = "../../../logs/test_onnx_eccv16_colorizer_3.jpg";
  std::string save_eccv_img_path4 = "../../../logs/test_onnx_eccv16_colorizer_one_piece_0.jpg";
  std::string save_siggraph_img_path1 = "../../../logs/test_onnx_siggraph17_colorizer_1.jpg";
  std::string save_siggraph_img_path2 = "../../../logs/test_onnx_siggraph17_colorizer_2.jpg";
  std::string save_siggraph_img_path3 = "../../../logs/test_onnx_siggraph17_colorizer_3.jpg";
  std::string save_siggraph_img_path4 = "../../../logs/test_onnx_siggraph17_colorizer_one_piece_0.jpg";

  lite::onnxruntime::cv::colorization::Colorizer *eccv16_colorizer =
      new lite::onnxruntime::cv::colorization::Colorizer(eccv16_onnx_path);
  lite::onnxruntime::cv::colorization::Colorizer *siggraph17_colorizer =
      new lite::onnxruntime::cv::colorization::Colorizer(siggraph17_onnx_path);

  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  cv::Mat img_bgr2 = cv::imread(test_img_path2);
  cv::Mat img_bgr3 = cv::imread(test_img_path3);
  cv::Mat img_bgr4 = cv::imread(test_img_path4);

  lite::onnxruntime::cv::types::ColorizeContent eccv16_colorize_content1;
  lite::onnxruntime::cv::types::ColorizeContent eccv16_colorize_content2;
  lite::onnxruntime::cv::types::ColorizeContent eccv16_colorize_content3;
  lite::onnxruntime::cv::types::ColorizeContent eccv16_colorize_content4;

  lite::onnxruntime::cv::types::ColorizeContent siggraph17_colorize_content1;
  lite::onnxruntime::cv::types::ColorizeContent siggraph17_colorize_content2;
  lite::onnxruntime::cv::types::ColorizeContent siggraph17_colorize_content3;
  lite::onnxruntime::cv::types::ColorizeContent siggraph17_colorize_content4;

  eccv16_colorizer->detect(img_bgr1, eccv16_colorize_content1);
  eccv16_colorizer->detect(img_bgr2, eccv16_colorize_content2);
  eccv16_colorizer->detect(img_bgr3, eccv16_colorize_content3);
  eccv16_colorizer->detect(img_bgr4, eccv16_colorize_content4);

  siggraph17_colorizer->detect(img_bgr1, siggraph17_colorize_content1);
  siggraph17_colorizer->detect(img_bgr2, siggraph17_colorize_content2);
  siggraph17_colorizer->detect(img_bgr3, siggraph17_colorize_content3);
  siggraph17_colorizer->detect(img_bgr4, siggraph17_colorize_content4);

  if (eccv16_colorize_content1.flag) cv::imwrite(save_eccv_img_path1, eccv16_colorize_content1.mat);
  if (eccv16_colorize_content2.flag) cv::imwrite(save_eccv_img_path2, eccv16_colorize_content2.mat);
  if (eccv16_colorize_content3.flag) cv::imwrite(save_eccv_img_path3, eccv16_colorize_content3.mat);
  if (eccv16_colorize_content4.flag) cv::imwrite(save_eccv_img_path4, eccv16_colorize_content4.mat);

  if (siggraph17_colorize_content1.flag) cv::imwrite(save_siggraph_img_path1, siggraph17_colorize_content1.mat);
  if (siggraph17_colorize_content2.flag) cv::imwrite(save_siggraph_img_path2, siggraph17_colorize_content2.mat);
  if (siggraph17_colorize_content3.flag) cv::imwrite(save_siggraph_img_path3, siggraph17_colorize_content3.mat);
  if (siggraph17_colorize_content4.flag) cv::imwrite(save_siggraph_img_path4, siggraph17_colorize_content4.mat);

  std::cout << "ONNXRuntime Version Colorization Done." << std::endl;

  delete eccv16_colorizer;
  delete siggraph17_colorizer;
}

static void test_mnn()
{

}

static void test_ncnn()
{

}

static void test_tnn()
{

}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}

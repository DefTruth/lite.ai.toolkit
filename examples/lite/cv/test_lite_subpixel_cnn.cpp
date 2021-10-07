//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/subpixel-cnn.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_subpixel_cnn.jpg";
  std::string save_img_path = "../../../logs/test_lite_subpixel_cnn.jpg";

  lite::cv::resolution::SubPixelCNN *subpixel_cnn = new lite::cv::resolution::SubPixelCNN(onnx_path);

  lite::types::SuperResolutionContent super_resolution_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  subpixel_cnn->detect(img_bgr, super_resolution_content);

  if (super_resolution_content.flag) cv::imwrite(save_img_path, super_resolution_content.mat);

  std::cout << "Default Version Done! Super Resolution Done." << std::endl;

  delete subpixel_cnn;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/subpixel-cnn.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_subpixel_cnn.jpg";
  std::string save_img_path = "../../../logs/test_onnx_subpixel_cnn.jpg";

  lite::onnxruntime::cv::resolution::SubPixelCNN *subpixel_cnn =
      new lite::onnxruntime::cv::resolution::SubPixelCNN(onnx_path);

  lite::types::SuperResolutionContent super_resolution_content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  subpixel_cnn->detect(img_bgr, super_resolution_content);

  if (super_resolution_content.flag) cv::imwrite(save_img_path, super_resolution_content.mat);

  std::cout << "ONNXRuntime Version Done! Super Resolution Done." << std::endl;

  delete subpixel_cnn;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
#endif
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

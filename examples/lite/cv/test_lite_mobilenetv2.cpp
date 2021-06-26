//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/mobilenetv2.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mobilenetv2.jpg";

  lite::cv::classification::MobileNetV2 *mobilenetv2 = new lite::cv::classification::MobileNetV2(onnx_path);

  lite::cv::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "Default Version Done!" << std::endl;
  }

  delete mobilenetv2;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/mobilenetv2.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_mobilenetv2.jpg";

  lite::onnxruntime::cv::classification::MobileNetV2 *mobilenetv2 =
      new lite::onnxruntime::cv::classification::MobileNetV2(onnx_path);

  lite::onnxruntime::cv::types::ImageNetContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2->detect(img_bgr, content);

  if (content.flag)
  {
    const unsigned int top_k = content.scores.size();
    if (top_k > 0)
    {
      for (unsigned int i = 0; i < top_k; ++i)
        std::cout << i + 1
                  << ": " << content.labels.at(i)
                  << ": " << content.texts.at(i)
                  << ": " << content.scores.at(i)
                  << std::endl;
    }
    std::cout << "ONNXRuntime Version Done!" << std::endl;
  }

  delete mobilenetv2;
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

//
// Created by DefTruth on 2021/6/24.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/age_googlenet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet.jpg";

  // 1. Test Default Engine ONNXRuntime
  lite::cv::face::attr::AgeGoogleNet *age_googlenet = new lite::cv::face::attr::AgeGoogleNet(onnx_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  if (age.flag)
    std::cout << "Default Version Detected Age: " << age.age << std::endl;

  delete age_googlenet;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/age_googlenet.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet.jpg";

  // 2. Test Specific Engine ONNXRuntime
  lite::onnxruntime::cv::face::attr::AgeGoogleNet *onnx_age_googlenet =
      new lite::onnxruntime::cv::face::attr::AgeGoogleNet(onnx_path);
  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  onnx_age_googlenet->detect(img_bgr, age);

  if (age.flag)
    std::cout << "ONNXRuntime Version Detected Age: " << age.age << std::endl;

  delete onnx_age_googlenet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/age_googlenet.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet_mnn.jpg";

  lite::mnn::cv::face::attr::AgeGoogleNet *age_googlenet =
      new lite::mnn::cv::face::attr::AgeGoogleNet(mnn_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  if (age.flag)
    std::cout << "MNN Version Detected Age: " << age.age << std::endl;

  delete age_googlenet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/age_googlenet.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/age_googlenet.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet_ncnn.jpg";

  lite::ncnn::cv::face::attr::AgeGoogleNet *age_googlenet =
      new lite::ncnn::cv::face::attr::AgeGoogleNet(param_path, bin_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  if (age.flag)
    std::cout << "NCNN Version Detected Age: " << age.age << std::endl;

  delete age_googlenet;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/age_googlenet.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/age_googlenet.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_age_googlenet.jpg";
  std::string save_img_path = "../../../logs/test_lite_age_googlenet_tnn.jpg";

  lite::tnn::cv::face::attr::AgeGoogleNet *age_googlenet =
      new lite::tnn::cv::face::attr::AgeGoogleNet(proto_path, model_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  age_googlenet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  if (age.flag)
    std::cout << "TNN Version Detected Age: " << age.age << std::endl;

  delete age_googlenet;
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

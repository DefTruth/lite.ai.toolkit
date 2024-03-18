//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/vgg_ilsvrc_16_gender_imdb_wiki.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_vgg16_gender.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_vgg16_gender.jpg";

  lite::cv::face::attr::VGG16Gender *vgg16_gender = new lite::cv::face::attr::VGG16Gender(onnx_path);

  lite::types::Gender gender;
  cv::Mat img_bgr = cv::imread(test_img_path);
  vgg16_gender->detect(img_bgr, gender);

  lite::utils::draw_gender_inplace(img_bgr, gender);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Gender: " << gender.label << std::endl;

  delete vgg16_gender;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/vgg_ilsvrc_16_gender_imdb_wiki.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_vgg16_gender.jpg";
  std::string save_img_path = "../../../examples/logs/test_onnx_vgg16_gender.jpg";

  lite::onnxruntime::cv::face::attr::VGG16Gender *vgg16_gender =
      new lite::onnxruntime::cv::face::attr::VGG16Gender(onnx_path);

  lite::types::Gender gender;
  cv::Mat img_bgr = cv::imread(test_img_path);
  vgg16_gender->detect(img_bgr, gender);

  lite::utils::draw_gender_inplace(img_bgr, gender);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Gender: " << gender.label << std::endl;

  delete vgg16_gender;
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

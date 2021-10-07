//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../logs/test_lite_pfld.jpg";

  lite::cv::face::align::PFLD *pfld = new lite::cv::face::align::PFLD(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../logs/test_onnx_pfld.jpg";

  lite::onnxruntime::cv::face::align::PFLD *pfld =
      new lite::onnxruntime::cv::face::align::PFLD(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld;
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

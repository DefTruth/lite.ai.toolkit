//
// Created by DefTruth on 2021/7/27.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/pytorch_face_landmarks_pfld.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_lite_pfld68.jpg";

  lite::cv::face::align::PFLD68 *pfld68 = new lite::cv::face::align::PFLD68(onnx_path);

  lite::cv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld68->detect(img_bgr, landmarks);

  lite::cv::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld68;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/pytorch_face_landmarks_pfld.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_onnx_pfld68.jpg";

  lite::onnxruntime::cv::face::align::PFLD68 *pfld68 =
      new lite::onnxruntime::cv::face::align::PFLD68(onnx_path);

  lite::onnxruntime::cv::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld68->detect(img_bgr, landmarks);

  lite::onnxruntime::cv::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld68;
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

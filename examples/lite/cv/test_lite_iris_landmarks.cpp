//
// Created by DefTruth on 2022/5/19.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/iris_landmark.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_screen_lefteye.png";
  std::string save_img_path = "../../../logs/test_lite_iris_landmarks_screen_lefteye.jpg";

  lite::cv::face::align3d::IrisLandmarks *iris_landmarks =
      new lite::cv::face::align3d::IrisLandmarks(onnx_path);

  const bool is_screen_right_eye = false;
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::Landmarks3D eyes_contours_and_brows, iris;
  iris_landmarks->detect(img_bgr, eyes_contours_and_brows, iris, is_screen_right_eye);

  lite::utils::draw_landmarks3d_inplace(img_bgr, iris);
  lite::utils::draw_landmarks3d_inplace(img_bgr, eyes_contours_and_brows);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Eye Contour and Brows Landmarks Num: "
            << eyes_contours_and_brows.points.size() << " Iris Landmarks Num: "
            << iris.points.size() << std::endl;

  delete iris_landmarks;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/iris_landmark.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_screen_righteye.png";
  std::string save_img_path = "../../../logs/test_lite_iris_landmarks_screen_righteye_onnx.jpg";

  lite::onnxruntime::cv::face::align3d::IrisLandmarks *iris_landmarks =
      new lite::onnxruntime::cv::face::align3d::IrisLandmarks(onnx_path);

  const bool is_screen_right_eye = true;
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::Landmarks3D eyes_contours_and_brows, iris;
  iris_landmarks->detect(img_bgr, eyes_contours_and_brows, iris, is_screen_right_eye);

  lite::utils::draw_landmarks3d_inplace(img_bgr, iris);
  lite::utils::draw_landmarks3d_inplace(img_bgr, eyes_contours_and_brows);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Eye Contour and Brows Landmarks Num: "
            << eyes_contours_and_brows.points.size() << " Iris Landmarks Num: "
            << iris.points.size() << std::endl;

  delete iris_landmarks;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/iris_landmark.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_screen_righteye.png";
  std::string save_img_path = "../../../logs/test_lite_iris_landmarks_screen_righteye_mnn.jpg";

  lite::mnn::cv::face::align3d::IrisLandmarks *iris_landmarks =
      new lite::mnn::cv::face::align3d::IrisLandmarks(mnn_path);

  const bool is_screen_right_eye = true;
  cv::Mat img_bgr = cv::imread(test_img_path);
  lite::types::Landmarks3D eyes_contours_and_brows, iris;
  iris_landmarks->detect(img_bgr, eyes_contours_and_brows, iris, is_screen_right_eye);

  lite::utils::draw_landmarks3d_inplace(img_bgr, iris);
  lite::utils::draw_landmarks3d_inplace(img_bgr, eyes_contours_and_brows);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Eye Contour and Brows Landmarks Num: "
            << eyes_contours_and_brows.points.size() << " Iris Landmarks Num: "
            << iris.points.size() << std::endl;

  delete iris_landmarks;
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

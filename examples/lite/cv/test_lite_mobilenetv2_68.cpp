//
// Created by DefTruth on 2021/7/27.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/pytorch_face_landmarks_landmark_detection_56.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../examples/logs/test_lite_mobilenetv2_68.jpg";

  lite::cv::face::align::MobileNetV268 *mobilenetv2_68 =
      new lite::cv::face::align::MobileNetV268(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2_68->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete mobilenetv2_68;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/pytorch_face_landmarks_landmark_detection_56.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../examples/logs/test_mobilenetv2_68_onnx.jpg";

  lite::onnxruntime::cv::face::align::MobileNetV268 *mobilenetv2_68 =
      new lite::onnxruntime::cv::face::align::MobileNetV268(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2_68->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete mobilenetv2_68;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/pytorch_face_landmarks_landmark_detection_56.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../examples/logs/test_mobilenetv2_68_mnn.jpg";

  lite::mnn::cv::face::align::MobileNetV268 *mobilenetv2_68 =
      new lite::mnn::cv::face::align::MobileNetV268(mnn_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2_68->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete mobilenetv2_68;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/pytorch_face_landmarks_landmark_detection_56.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/pytorch_face_landmarks_landmark_detection_56.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../examples/logs/test_mobilenetv2_68_ncnn.jpg";

  lite::ncnn::cv::face::align::MobileNetV268 *mobilenetv2_68 =
      new lite::ncnn::cv::face::align::MobileNetV268(param_path, bin_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2_68->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete mobilenetv2_68;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/pytorch_face_landmarks_landmark_detection_56.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/pytorch_face_landmarks_landmark_detection_56.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../examples/logs/test_mobilenetv2_68_tnn.jpg";

  lite::tnn::cv::face::align::MobileNetV268 *mobilenetv2_68 =
      new lite::tnn::cv::face::align::MobileNetV268(proto_path, model_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  mobilenetv2_68->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete mobilenetv2_68;
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

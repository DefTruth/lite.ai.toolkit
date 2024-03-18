//
// Created by DefTruth on 2022/3/23.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/pipnet_resnet18_10x19x32x256_aflw.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pipnet19.jpg";

  lite::cv::face::align::PIPNet19 *pipnet19 = new lite::cv::face::align::PIPNet19(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pipnet19->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pipnet19;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/pipnet_resnet18_10x19x32x256_aflw.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pipnet19_onnx.jpg";

  lite::onnxruntime::cv::face::align::PIPNet19 *pipnet19 =
      new lite::onnxruntime::cv::face::align::PIPNet19(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pipnet19->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pipnet19;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/pipnet_resnet18_10x19x32x256_aflw.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pipnet19_mnn.jpg";

  lite::mnn::cv::face::align::PIPNet19 *pipnet19 =
      new lite::mnn::cv::face::align::PIPNet19(mnn_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pipnet19->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pipnet19;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/pipnet_resnet18_10x19x32x256_aflw.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/pipnet_resnet18_10x19x32x256_aflw.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pipnet19_ncnn.jpg";

  lite::ncnn::cv::face::align::PIPNet19 *pipnet19 =
      new lite::ncnn::cv::face::align::PIPNet19(param_path, bin_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pipnet19->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pipnet19;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/pipnet_resnet18_10x19x32x256_aflw.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/pipnet_resnet18_10x19x32x256_aflw.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pipnet19_tnn.jpg";

  lite::tnn::cv::face::align::PIPNet19 *pipnet19 =
      new lite::tnn::cv::face::align::PIPNet19(proto_path, model_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pipnet19->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pipnet19;
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

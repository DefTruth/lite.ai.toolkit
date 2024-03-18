//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../examples/logs/test_lite_pfld.jpg";

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
  std::string onnx_path = "../../../examples/hub/onnx/cv/pfld-106-v3.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../examples/logs/test_pfld_onnx.jpg";

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
  std::string mnn_path = "../../../examples/hub/mnn/cv/pfld-106-v3.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../examples/logs/test_pfld_mnn.jpg";

  lite::mnn::cv::face::align::PFLD *pfld =
      new lite::mnn::cv::face::align::PFLD(mnn_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/pfld-106-v3.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/pfld-106-v3.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../examples/logs/test_pfld_ncnn.jpg";

  lite::ncnn::cv::face::align::PFLD *pfld =
      new lite::ncnn::cv::face::align::PFLD(param_path, bin_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/pfld-106-v3.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/pfld-106-v3.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_pfld.png";
  std::string save_img_path = "../../../examples/logs/test_pfld_tnn.jpg";

  lite::tnn::cv::face::align::PFLD *pfld =
      new lite::tnn::cv::face::align::PFLD(proto_path, model_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld;
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

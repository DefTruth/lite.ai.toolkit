//
// Created by DefTruth on 2021/7/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/PFLD-pytorch-pfld.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_lite_pfld98.jpg";

  lite::cv::face::align::PFLD98 *pfld98 = new lite::cv::face::align::PFLD98(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld98->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld98;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/PFLD-pytorch-pfld.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_pfld98_onnx.jpg";

  lite::onnxruntime::cv::face::align::PFLD98 *pfld98 =
      new lite::onnxruntime::cv::face::align::PFLD98(onnx_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld98->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld98;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/PFLD-pytorch-pfld.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_pfld98_mnn.jpg";

  lite::mnn::cv::face::align::PFLD98 *pfld98 =
      new lite::mnn::cv::face::align::PFLD98(mnn_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld98->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld98;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/PFLD-pytorch-pfld.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/PFLD-pytorch-pfld.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_pfld98_ncnn.jpg";

  lite::ncnn::cv::face::align::PFLD98 *pfld98 =
      new lite::ncnn::cv::face::align::PFLD98(param_path, bin_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld98->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld98;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/PFLD-pytorch-pfld.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/PFLD-pytorch-pfld.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_landmarks.png";
  std::string save_img_path = "../../../logs/test_pfld98_tnn.jpg";

  lite::tnn::cv::face::align::PFLD98 *pfld98 =
      new lite::tnn::cv::face::align::PFLD98(proto_path, model_path);

  lite::types::Landmarks landmarks;
  cv::Mat img_bgr = cv::imread(test_img_path);
  pfld98->detect(img_bgr, landmarks);

  lite::utils::draw_landmarks_inplace(img_bgr, landmarks);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected Landmarks Num: "
            << landmarks.points.size() << std::endl;

  delete pfld98;
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

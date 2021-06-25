//
// Created by DefTruth on 2021/6/25.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/arcfaceresnet100-8.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_ortcv_arcface_resnet_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_ortcv_arcface_resnet_1.png";

  lite::cv::faceid::ArcFaceResNet *arcface_resnet = new lite::cv::faceid::ArcFaceResNet(onnx_path);

  lite::cv::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  arcface_resnet->detect(img_bgr0, face_content0);
  arcface_resnet->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "Default Version Detected Sim: " << sim << std::endl;
  }

  delete arcface_resnet;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/arcfaceresnet100-8.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_ortcv_arcface_resnet_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_ortcv_arcface_resnet_1.png";

  lite::onnxruntime::cv::faceid::ArcFaceResNet *arcface_resnet =
      new lite::onnxruntime::cv::faceid::ArcFaceResNet(onnx_path);

  lite::onnxruntime::cv::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  arcface_resnet->detect(img_bgr0, face_content0);
  arcface_resnet->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::onnxruntime::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "ONNXRuntime Version Detected Sim: " << sim << std::endl;
  }

  delete arcface_resnet;
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
  //  Ort::Exception: Non-zero status code returned while running BatchNormalization node.
  //  Name:'stage1_unit1_bn1' Status Message:
  //  Invalid input scale: NumDimensions() != 3
}

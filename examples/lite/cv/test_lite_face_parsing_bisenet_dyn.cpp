//
// Created by DefTruth on 2022/7/2.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/face_parsing_dynamic.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_parsing.png";
  std::string save_img_path = "../../../logs/test_lite_face_parsing_bisenet_dyn.jpg";

  lite::cv::segmentation::FaceParsingBiSeNetDyn *face_parsing_bisenet_dyn =
      new lite::cv::segmentation::FaceParsingBiSeNetDyn(onnx_path, 4); // 4 threads

  lite::types::FaceParsingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_parsing_bisenet_dyn->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.merge.empty()) cv::imwrite(save_img_path, content.merge);
    std::cout << "Default Version FaceParsingBiSeNetDyn Done!" << std::endl;
  }

  delete face_parsing_bisenet_dyn;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/face_parsing_dynamic.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_parsing.png";
  std::string save_img_path = "../../../logs/test_lite_face_parsing_bisenet_dyn_onnx.jpg";

  lite::onnxruntime::cv::segmentation::FaceParsingBiSeNetDyn *face_parsing_bisenet_dyn =
      new lite::onnxruntime::cv::segmentation::FaceParsingBiSeNetDyn(onnx_path, 4); // 4 threads

  lite::types::FaceParsingContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  face_parsing_bisenet_dyn->detect(img_bgr, content);

  if (content.flag)
  {
    if (!content.merge.empty()) cv::imwrite(save_img_path, content.merge);
    std::cout << "ONNXRuntime Version FaceParsingBiSeNetDyn Done!" << std::endl;
  }

  delete face_parsing_bisenet_dyn;
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

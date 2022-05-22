//
// Created by DefTruth on 2022/5/19.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/facemesh_face_landmark.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align.png";
  std::string save_img_path = "../../../logs/test_lite_facemesh.jpg";

  lite::cv::face::align3d::FaceMesh *facemesh =
      new lite::cv::face::align3d::FaceMesh(onnx_path);

  float confidence = 0.f;
  lite::types::Landmarks3D landmarks3d;
  cv::Mat img_bgr = cv::imread(test_img_path);
  facemesh->detect(img_bgr, landmarks3d, confidence);

  lite::utils::draw_facemesh_inplace(img_bgr, landmarks3d);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected Landmarks Num: "
            << landmarks3d.points.size() << "Confidence: "
            << confidence << std::endl;

  delete facemesh;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/facemesh_face_landmark.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align_2.png";
  std::string save_img_path = "../../../logs/test_lite_facemesh_onnx.jpg";

  lite::onnxruntime::cv::face::align3d::FaceMesh *facemesh =
      new lite::onnxruntime::cv::face::align3d::FaceMesh(onnx_path);

  float confidence = 0.f;
  lite::types::Landmarks3D landmarks3d;
  cv::Mat img_bgr = cv::imread(test_img_path);
  facemesh->detect(img_bgr, landmarks3d, confidence);

  lite::utils::draw_facemesh_inplace(img_bgr, landmarks3d);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected Landmarks Num: "
            << landmarks3d.points.size() << " Confidence: "
            << confidence << std::endl;

  delete facemesh;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/facemesh_face_landmark.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_face_align_3.png";
  std::string save_img_path = "../../../logs/test_lite_facemesh_mnn.jpg";

  lite::mnn::cv::face::align3d::FaceMesh *facemesh =
      new lite::mnn::cv::face::align3d::FaceMesh(mnn_path);

  float confidence = 0.f;
  lite::types::Landmarks3D landmarks3d;
  cv::Mat img_bgr = cv::imread(test_img_path);
  facemesh->detect(img_bgr, landmarks3d, confidence);

  lite::utils::draw_facemesh_inplace(img_bgr, landmarks3d);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected Landmarks Num: "
            << landmarks3d.points.size() << " Confidence: "
            << confidence << std::endl;

  delete facemesh;
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

//
// Created by DefTruth on 2021/7/18.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/dream_ijba_res18_end2end.onnx";
  std::string pose_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_1.png";

  lite::cv::faceid::PoseRobustFace *pose_robust_face = new lite::cv::faceid::PoseRobustFace(onnx_path);
  lite::cv::face::pose::FSANet *fsanet = new lite::cv::face::pose::FSANet(pose_onnx_path);

  lite::cv::types::FaceContent face_content0, face_content1;
  lite::cv::types::EulerAngles euler_angles0, euler_angles1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  fsanet->detect(img_bgr0, euler_angles0); // head pose estimation first
  fsanet->detect(img_bgr1, euler_angles1);

  if (euler_angles0.flag && euler_angles1.flag)
  {
    // need a yaw value of input image, but can not get good performance
    // in my test. so, I set it 0 in order to get default and robust performance
    // pose_robust_face->detect(img_bgr0, face_content0, euler_angles0.yaw);
    // pose_robust_face->detect(img_bgr1, face_content1, euler_angles1.yaw);
    pose_robust_face->detect(img_bgr0, face_content0, 0.f);
    pose_robust_face->detect(img_bgr1, face_content1, 0.f);
  } else
  {
    pose_robust_face->detect(img_bgr0, face_content0, 0.f);
    pose_robust_face->detect(img_bgr1, face_content1, 0.f);
  }

  std::cout << "face0.yaw: " << euler_angles0.yaw << " "
            << "face1.yaw: " << euler_angles1.yaw << std::endl;

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "Default Version Detected Sim: " << sim << std::endl;
  }

  delete pose_robust_face;
  delete fsanet;
}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/dream_ijba_res18_end2end.onnx";
  std::string pose_onnx_path = "../../../hub/onnx/cv/fsanet-var.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::onnxruntime::cv::faceid::PoseRobustFace *pose_robust_face =
      new lite::onnxruntime::cv::faceid::PoseRobustFace(onnx_path);
  lite::onnxruntime::cv::face::pose::FSANet *fsanet =
      new lite::onnxruntime::cv::face::pose::FSANet(pose_onnx_path);

  lite::onnxruntime::cv::types::FaceContent face_content0, face_content1;
  lite::cv::types::EulerAngles euler_angles0, euler_angles1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  fsanet->detect(img_bgr0, euler_angles0); // head pose estimation first
  fsanet->detect(img_bgr1, euler_angles1);

  if (euler_angles0.flag && euler_angles1.flag)
  {
    // pose_robust_face->detect(img_bgr0, face_content0, euler_angles0.yaw);
    // pose_robust_face->detect(img_bgr1, face_content1, euler_angles1.yaw);
    pose_robust_face->detect(img_bgr0, face_content0, 0.f);
    pose_robust_face->detect(img_bgr1, face_content1, 0.f);
  } else
  {
    pose_robust_face->detect(img_bgr0, face_content0, 0.f);
    pose_robust_face->detect(img_bgr1, face_content1, 0.f);
  }
  std::cout << "face0.yaw: " << euler_angles0.yaw << " "
            << "face1.yaw: " << euler_angles1.yaw << std::endl;

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::onnxruntime::cv::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "ONNXRuntime Version Detected Sim: " << sim << std::endl;
  }

  delete pose_robust_face;
  delete fsanet;
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

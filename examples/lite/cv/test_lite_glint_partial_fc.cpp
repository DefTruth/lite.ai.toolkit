//
// Created by DefTruth on 2021/7/4.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/partial_fc_glint360k_r50.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_glint_partial_fc_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_glint_partial_fc_1.png";

  lite::cv::faceid::GlintPartialFC *glint_partial_fc = new lite::cv::faceid::GlintPartialFC(onnx_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  glint_partial_fc->detect(img_bgr0, face_content0);
  glint_partial_fc->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "Default Version Detected Sim: " << sim << std::endl;
  }

  delete glint_partial_fc;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/partial_fc_glint360k_r50.onnx";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_glint_partial_fc_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_glint_partial_fc_2.png";

  lite::onnxruntime::cv::faceid::GlintPartialFC *glint_partial_fc =
      new lite::onnxruntime::cv::faceid::GlintPartialFC(onnx_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  glint_partial_fc->detect(img_bgr0, face_content0);
  glint_partial_fc->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "ONNXRuntime Version Detected Sim: " << sim << std::endl;
  }

  delete glint_partial_fc;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/partial_fc_glint360k_r50.mnn";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_glint_partial_fc_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_glint_partial_fc_2.png";

  lite::mnn::cv::faceid::GlintPartialFC *glint_partial_fc =
      new lite::mnn::cv::faceid::GlintPartialFC(mnn_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  glint_partial_fc->detect(img_bgr0, face_content0);
  glint_partial_fc->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "MNN Version Detected Sim: " << sim << std::endl;
  }

  delete glint_partial_fc;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/partial_fc_glint360k_r50.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/partial_fc_glint360k_r50.opt.bin";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::ncnn::cv::faceid::GlintPartialFC *glint_partial_fc =
      new lite::ncnn::cv::faceid::GlintPartialFC(param_path, bin_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  glint_partial_fc->detect(img_bgr0, face_content0);
  glint_partial_fc->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "NCNN Version Detected Sim: " << sim << std::endl;
  }

  delete glint_partial_fc;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/partial_fc_glint360k_r50.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/partial_fc_glint360k_r50.opt.tnnmodel";
  std::string test_img_path0 = "../../../examples/lite/resources/test_lite_faceid_0.png";
  std::string test_img_path1 = "../../../examples/lite/resources/test_lite_faceid_2.png";

  lite::tnn::cv::faceid::GlintPartialFC *glint_partial_fc =
      new lite::tnn::cv::faceid::GlintPartialFC(proto_path, model_path);

  lite::types::FaceContent face_content0, face_content1;
  cv::Mat img_bgr0 = cv::imread(test_img_path0);
  cv::Mat img_bgr1 = cv::imread(test_img_path1);
  glint_partial_fc->detect(img_bgr0, face_content0);
  glint_partial_fc->detect(img_bgr1, face_content1);

  if (face_content0.flag && face_content1.flag)
  {
    float sim = lite::utils::math::cosine_similarity<float>(
        face_content0.embedding, face_content1.embedding);
    std::cout << "TNN Version Detected Sim: " << sim << std::endl;
  }

  delete glint_partial_fc;
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

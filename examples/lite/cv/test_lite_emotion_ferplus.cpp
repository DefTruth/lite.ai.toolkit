//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/emotion-ferplus-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../logs/test_lite_emotion_ferplus.jpg";

  lite::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::cv::face::attr::EmotionFerPlus(onnx_path);

  lite::cv::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::cv::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;

}

static void test_onnxruntime()
{
  std::string onnx_path = "../../../hub/onnx/cv/emotion-ferplus-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../logs/test_lite_emotion_ferplus.jpg";

  lite::onnxruntime::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::onnxruntime::cv::face::attr::EmotionFerPlus(onnx_path);

  lite::onnxruntime::cv::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::onnxruntime::cv::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;
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

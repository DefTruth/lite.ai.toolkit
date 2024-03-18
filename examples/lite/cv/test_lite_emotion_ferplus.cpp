//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/emotion-ferplus-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_emotion_ferplus.jpg";

  lite::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::cv::face::attr::EmotionFerPlus(onnx_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;

}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/emotion-ferplus-8.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_emotion_ferplus.jpg";

  lite::onnxruntime::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::onnxruntime::cv::face::attr::EmotionFerPlus(onnx_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/emotion-ferplus-8.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_emotion_ferplus_mnn.jpg";

  lite::mnn::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::mnn::cv::face::attr::EmotionFerPlus(mnn_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/emotion-ferplus-8.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/emotion-ferplus-8.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_emotion_ferplus_ncnn.jpg";

  lite::ncnn::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::ncnn::cv::face::attr::EmotionFerPlus(param_path, bin_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "NCNN Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/emotion-ferplus-8.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/emotion-ferplus-8.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_emotion_ferplus.jpg";
  std::string save_img_path = "../../../examples/logs/test_lite_emotion_ferplus_tnn.jpg";

  lite::tnn::cv::face::attr::EmotionFerPlus *emotion_ferplus =
      new lite::tnn::cv::face::attr::EmotionFerPlus(proto_path, model_path);

  lite::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  lite::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;
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

//
// Created by DefTruth on 2021/4/3.
//

#include <iostream>
#include <vector>
#include "ort/cv/emotion_ferplus.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_emotion_ferplus()
{
  std::string onnx_path = "../../../hub/onnx/cv/emotion-ferplus-8.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_emotion_ferplus.jpg";
  std::string save_img_path = "../../../logs/test_ortcv_emotion_ferplus.jpg";

  ortcv::EmotionFerPlus *emotion_ferplus = new ortcv::EmotionFerPlus(onnx_path);

  ortcv::types::Emotions emotions;
  cv::Mat img_bgr = cv::imread(test_img_path);
  emotion_ferplus->detect(img_bgr, emotions);

  ortcv::utils::draw_emotion_inplace(img_bgr, emotions);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Emotion: " << emotions.text << std::endl;

  delete emotion_ferplus;

}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_emotion_ferplus();
  return 0;
}
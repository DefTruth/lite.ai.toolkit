//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/fcn_resnet101.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fcn_resnet101.png";
  std::string save_img_path = "../../../logs/test_lite_fcn_resnet101.jpg";

  lite::cv::segmentation::FCNResNet101 *fcn_resnet101 =
      new lite::cv::segmentation::FCNResNet101(onnx_path, 16);

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fcn_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat,0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "Detected Label: " << it->first << " Name: " << it->second << std::endl;
      }
    }
    std::cout << "Default Version Done!" << std::endl;
  }

  delete fcn_resnet101;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/fcn_resnet101.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_fcn_resnet101.png";
  std::string save_img_path = "../../../logs/test_onnx_fcn_resnet101.jpg";

  lite::onnxruntime::cv::segmentation::FCNResNet101 *fcn_resnet101 =
      new lite::onnxruntime::cv::segmentation::FCNResNet101(onnx_path, 16);

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  fcn_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat,0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "Detected Label: " << it->first << " Name: " << it->second << std::endl;
      }
    }
    std::cout << "ONNXRuntime Version Done!" << std::endl;
  }

  delete fcn_resnet101;
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

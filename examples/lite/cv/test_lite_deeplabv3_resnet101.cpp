//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_lite_deeplabv3_resnet101.jpg";

  lite::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "Default Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_onnx_deeplabv3_resnet101.jpg";

  lite::onnxruntime::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::onnxruntime::cv::segmentation::DeepLabV3ResNet101(onnx_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "ONNXRuntime Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../hub/mnn/cv/deeplabv3_resnet101_coco.mnn";
  // std::string mnn_path = "../../../hub/onnx/cv/deeplabv3_resnet101_coco.opt.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_deeplabv3_resnet101_mnn.jpg";

  lite::mnn::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::mnn::cv::segmentation::DeepLabV3ResNet101(mnn_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "MNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../hub/ncnn/cv/deeplabv3_resnet101_coco.opt.param";
  std::string bin_path = "../../../hub/ncnn/cv/deeplabv3_resnet101_coco.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_deeplabv3_resnet101_ncnn.jpg";

  lite::ncnn::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::ncnn::cv::segmentation::DeepLabV3ResNet101(param_path, bin_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "NCNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../hub/tnn/cv/deeplabv3_resnet101_coco.opt.tnnproto";
  std::string model_path = "../../../hub/tnn/cv/deeplabv3_resnet101_coco.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_deeplabv3_resnet101.png";
  std::string save_img_path = "../../../logs/test_deeplabv3_resnet101_tnn.jpg";

  lite::tnn::cv::segmentation::DeepLabV3ResNet101 *deeplabv3_resnet101 =
      new lite::tnn::cv::segmentation::DeepLabV3ResNet101(proto_path, model_path, 16); // 16 threads

  lite::types::SegmentContent content;
  cv::Mat img_bgr = cv::imread(test_img_path);
  deeplabv3_resnet101->detect(img_bgr, content);

  if (content.flag)
  {
    cv::Mat out_img;
    cv::addWeighted(img_bgr, 0.2, content.color_mat, 0.8, 0., out_img);
    cv::imwrite(save_img_path, out_img);
    if (!content.names_map.empty())
    {
      for (auto it = content.names_map.begin(); it != content.names_map.end(); ++it)
      {
        std::cout << "TNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }
    }
  }

  delete deeplabv3_resnet101;
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

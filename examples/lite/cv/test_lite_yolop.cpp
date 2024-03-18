//
// Created by DefTruth on 2021/9/14.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolop-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../examples/logs/test_lite_yolop_det.jpg";
  std::string save_da_path = "../../../examples/logs/test_lite_yolop_da.jpg";
  std::string save_ll_path = "../../../examples/logs/test_lite_yolop_ll.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_yolop_merge.jpg";

  lite::cv::detection::YOLOP *yolop = new lite::cv::detection::YOLOP(onnx_path, 16); // 16 threads

  lite::types::SegmentContent da_seg_content;
  lite::types::SegmentContent ll_seg_content;
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "Default Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "Default Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../../../examples/hub/onnx/cv/yolop-640-640.onnx";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../examples/logs/test_lite_yolop_det_onnx.jpg";
  std::string save_da_path = "../../../examples/logs/test_lite_yolop_da_onnx.jpg";
  std::string save_ll_path = "../../../examples/logs/test_lite_yolop_ll_onnx.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_yolop_merge_onnx.jpg";

  lite::onnxruntime::cv::detection::YOLOP *yolop =
      new lite::onnxruntime::cv::detection::YOLOP(onnx_path, 16); // 16 threads

  lite::types::SegmentContent da_seg_content;
  lite::types::SegmentContent ll_seg_content;
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "ONNXRuntime Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "ONNXRuntime Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../../../examples/hub/mnn/cv/yolop-640-640.mnn";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../examples/logs/test_lite_yolop_det_mnn.jpg";
  std::string save_da_path = "../../../examples/logs/test_lite_yolop_da_mnn.jpg";
  std::string save_ll_path = "../../../examples/logs/test_lite_yolop_ll_mnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_yolop_merge_mnn.jpg";

  lite::mnn::cv::detection::YOLOP *yolop =
      new lite::mnn::cv::detection::YOLOP(mnn_path, 16); // 16 threads

  lite::types::SegmentContent da_seg_content;
  lite::types::SegmentContent ll_seg_content;
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "MNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "MNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
  std::string param_path = "../../../examples/hub/ncnn/cv/yolop-640-640.opt.param";
  std::string bin_path = "../../../examples/hub/ncnn/cv/yolop-640-640.opt.bin";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../examples/logs/test_lite_yolop_det_ncnn.jpg";
  std::string save_da_path = "../../../examples/logs/test_lite_yolop_da_ncnn.jpg";
  std::string save_ll_path = "../../../examples/logs/test_lite_yolop_ll_ncnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_yolop_merge_ncnn.jpg";

  lite::ncnn::cv::detection::YOLOP *yolop =
      new lite::ncnn::cv::detection::YOLOP(param_path, bin_path, 16); // 16 threads

  lite::types::SegmentContent da_seg_content;
  lite::types::SegmentContent ll_seg_content;
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "NCNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "NCNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../../../examples/hub/tnn/cv/yolop-640-640.opt.tnnproto";
  std::string model_path = "../../../examples/hub/tnn/cv/yolop-640-640.opt.tnnmodel";
  std::string test_img_path = "../../../examples/lite/resources/test_lite_yolop.jpg";
  std::string save_det_path = "../../../examples/logs/test_lite_yolop_det_tnn.jpg";
  std::string save_da_path = "../../../examples/logs/test_lite_yolop_da_tnn.jpg";
  std::string save_ll_path = "../../../examples/logs/test_lite_yolop_ll_tnn.jpg";
  std::string save_merge_path = "../../../examples/logs/test_lite_yolop_merge_tnn.jpg";

  lite::tnn::cv::detection::YOLOP *yolop =
      new lite::tnn::cv::detection::YOLOP(proto_path, model_path, 16); // 16 threads

  lite::types::SegmentContent da_seg_content;
  lite::types::SegmentContent ll_seg_content;
  std::vector<lite::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

  if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag)
  {
    // boxes.
    cv::Mat img_det = img_bgr.clone();
    lite::utils::draw_boxes_inplace(img_det, detected_boxes);
    cv::imwrite(save_det_path, img_det);
    std::cout << "Saved " << save_det_path << " done!" << "\n";
    // da && ll seg
    cv::imwrite(save_da_path, da_seg_content.class_mat);
    cv::imwrite(save_ll_path, ll_seg_content.class_mat);
    std::cout << "Saved " << save_da_path << " done!" << "\n";
    std::cout << "Saved " << save_ll_path << " done!" << "\n";
    // merge
    cv::Mat img_merge = img_bgr.clone();
    cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;

    cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
    lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
    cv::imwrite(save_merge_path, img_merge);
    std::cout << "Saved " << save_merge_path << " done!" << "\n";

    // label
    if (!da_seg_content.names_map.empty() && !ll_seg_content.names_map.empty())
    {

      for (auto it = da_seg_content.names_map.begin(); it != da_seg_content.names_map.end(); ++it)
      {
        std::cout << "TNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

      for (auto it = ll_seg_content.names_map.begin(); it != ll_seg_content.names_map.end(); ++it)
      {
        std::cout << "TNN Version Detected Label: "
                  << it->first << " Name: " << it->second << std::endl;
      }

    }
  }

  delete yolop;
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

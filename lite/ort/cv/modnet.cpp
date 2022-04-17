//
// Created by DefTruth on 2022/3/27.
//

#include "modnet.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::MODNet;

Ort::Value MODNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32
  // e.g (1,3,512,512)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // deepcopy inside
}


void MODNet::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise,
                    bool minimum_post_process)
{
  if (mat.empty()) return;

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise, minimum_post_process);
}

void MODNet::generate_matting(std::vector<Ort::Value> &output_tensors,
                              const cv::Mat &mat, types::MattingContent &content,
                              bool remove_noise, bool minimum_post_process)
{
  Ort::Value &output = output_tensors.at(0); // (1,1,h,w) 0~1
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);

  float *output_ptr = output.GetTensorMutableData<float>();

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, output_ptr);
  // post process
  if (remove_noise) lite::utils::remove_small_connected_area(alpha_pred, 0.05f);
  // resize alpha
  if (out_h != h || out_w != w)
  {
    cv::resize(alpha_pred, alpha_pred, cv::Size(w, h));
  }
  cv::Mat pmat = alpha_pred; // ref
  content.pha_mat = pmat;

  if (!minimum_post_process)
  {
    // MODNet only predict Alpha, no fgr. So,
    // the fake fgr and merge mat may not need,
    // let the fgr mat and merge mat empty to
    // Speed up the post processes.
    cv::Mat mat_copy;
    mat.convertTo(mat_copy, CV_32FC3);
    // merge mat and fgr mat may not need
    std::vector<cv::Mat> mat_channels;
    cv::split(mat_copy, mat_channels);
    cv::Mat bmat = mat_channels.at(0);
    cv::Mat gmat = mat_channels.at(1);
    cv::Mat rmat = mat_channels.at(2); // ref only, zero-copy.
    bmat = bmat.mul(pmat);
    gmat = gmat.mul(pmat);
    rmat = rmat.mul(pmat);
    cv::Mat rest = 1.f - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;
    std::vector<cv::Mat> fgr_channel_mats, merge_channel_mats;
    fgr_channel_mats.push_back(bmat);
    fgr_channel_mats.push_back(gmat);
    fgr_channel_mats.push_back(rmat);
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);

    cv::merge(fgr_channel_mats, content.fgr_mat);
    cv::merge(merge_channel_mats, content.merge_mat);

    content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  content.flag = true;
}

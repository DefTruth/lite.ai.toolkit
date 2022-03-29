//
// Created by DefTruth on 2022/3/27.
//

#include "modnet.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::MODNet;

Ort::Value MODNet::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32
  // e.g (1,3,224,224)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // deepcopy inside
}


void MODNet::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise)
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
  this->generate_matting(output_tensors, mat, content, remove_noise);
}

// user-friendly method for swap background.
void MODNet::swap_background(const cv::Mat &fg_mat,
                             const cv::Mat &pha_mat,
                             const cv::Mat &bg_mat,
                             cv::Mat &out_mat)
{
  if (fg_mat.empty() || pha_mat.empty() || bg_mat.empty()) return;
  const unsigned int fg_h = fg_mat.rows;
  const unsigned int fg_w = fg_mat.cols;
  const unsigned int bg_h = bg_mat.rows;
  const unsigned int bg_w = bg_mat.cols;
  const unsigned int ph_h = pha_mat.rows;
  const unsigned int ph_w = pha_mat.cols;
  cv::Mat bg_mat_copy, pha_mat_copy, fg_mat_copy;
  if (bg_h != fg_h || bg_w != fg_w)
    cv::resize(bg_mat, bg_mat_copy, cv::Size(fg_w, fg_h));
  else bg_mat_copy = bg_mat; // ref only.
  if (ph_h != fg_h || ph_w != fg_w)
    cv::resize(pha_mat, pha_mat_copy, cv::Size(fg_w, fg_h));
  else pha_mat_copy = pha_mat; // ref only.
  // convert pha_mat_copy to 3 channels.
  if (pha_mat_copy.channels() == 1)
    cv::cvtColor(pha_mat_copy, pha_mat_copy, cv::COLOR_GRAY2BGR); // 0.~1.
  // convert mats to float32 points.
  fg_mat.convertTo(fg_mat_copy, CV_32FC3); // 0.~255.
  bg_mat_copy.convertTo(bg_mat_copy, CV_32FC3); // 0.~255.
  pha_mat_copy.convertTo(pha_mat_copy, CV_32FC3); // 0.~1. assert pha_mat_copy is float32.
  // element wise operations.
  cv::Mat rest = 1.f - pha_mat_copy;
  // out_mat = fg_mat_copy.mul(pha_mat_copy) + bg_mat_copy.mul(rest);
  cv::add(fg_mat_copy.mul(pha_mat_copy), bg_mat_copy.mul(rest), out_mat);
  // check with 'out_mat.empty()' and CV_8UC3.
  if (!out_mat.empty()) out_mat.convertTo(out_mat, CV_8UC3);
}

void MODNet::generate_matting(std::vector<Ort::Value> &output_tensors,
                              const cv::Mat &mat, types::MattingContent &content,
                              bool remove_noise)
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
  if (remove_noise) this->remove_small_connected_area(alpha_pred);
  // resize alpha
  if (out_h != h || out_w != w)
  {
    cv::resize(alpha_pred, alpha_pred, cv::Size(w, h));
  }
  cv::Mat mat_copy;
  mat.convertTo(mat_copy, CV_32FC3);
  cv::Mat pmat = alpha_pred; // ref

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

  content.pha_mat = pmat;
  cv::merge(fgr_channel_mats, content.fgr_mat);
  cv::merge(merge_channel_mats, content.merge_mat);

  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);
  content.merge_mat.convertTo(content.merge_mat, CV_8UC3);

  content.flag = true;
}

// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L208
void MODNet::remove_small_connected_area(cv::Mat &alpha_pred)
{
  cv::Mat gray, binary;
  alpha_pred.convertTo(gray, CV_8UC1, 255.f);
  // 255 * 0.05 ~ 13
  // https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L209
  cv::threshold(gray, binary, 13, 255, cv::THRESH_BINARY);
  // morphologyEx with OPEN operation to remove noise first.
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(-1, -1));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  // Computationally connected domain
  cv::Mat labels = cv::Mat::zeros(alpha_pred.size(), CV_32S);
  cv::Mat stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
  if (num_labels <= 1) return; // no noise, skip.
  // find max connected area, 0 is background
  int max_connected_id = 1; // 1,2,...
  int max_connected_area = stats.at<int>(max_connected_id, cv::CC_STAT_AREA);
  for (int i = 1; i < num_labels; ++i)
  {
    int tmp_connected_area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (tmp_connected_area > max_connected_area)
    {
      max_connected_area = tmp_connected_area;
      max_connected_id = i;
    }
  }
  const int h = alpha_pred.rows;
  const int w = alpha_pred.cols;
  // remove small connected area.
  for (int i = 0; i < h; ++i)
  {
    int *label_row_ptr = labels.ptr<int>(i);
    float *alpha_row_ptr = alpha_pred.ptr<float>(i);
    for (int j = 0; j < w; ++j)
    {
      if (label_row_ptr[j] != max_connected_id)
        alpha_row_ptr[j] = 0.f;
    }
  }
}
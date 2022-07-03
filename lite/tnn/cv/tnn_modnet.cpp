//
// Created by DefTruth on 2022/3/27.
//

#include "tnn_modnet.h"
#include "lite/utils.h"

using tnncv::TNNMODNet;


TNNMODNet::TNNMODNet(const std::string &_proto_path,
                     const std::string &_model_path,
                     unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNMODNet::transform(const cv::Mat &mat_rs)
{
  // push into input_mat (1,3,512,512)
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNMODNet::detect(const cv::Mat &mat, types::MattingContent &content, bool remove_noise,
                       bool minimum_post_process)
{
  if (mat.empty()) return;

  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  this->transform(mat_rs);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. generate matting
  this->generate_matting(instance, mat, content, remove_noise, minimum_post_process);
}

void TNNMODNet::generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                                 const cv::Mat &mat, types::MattingContent &content,
                                 bool remove_noise, bool minimum_post_process)
{
  std::shared_ptr<tnn::Mat> output_mat;
  tnn::MatConvertParam cvt_param;
  auto status = _instance->GetOutputMat(output_mat, cvt_param, "output", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = output_mat->GetDims();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);

  float *output_ptr = (float *) output_mat->GetData();

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, output_ptr);
  if (remove_noise) lite::utils::remove_small_connected_area(alpha_pred, 0.05f);
  // resize alpha
  if (out_h != h || out_w != w)
    // already allocated a new continuous memory after resize.
    cv::resize(alpha_pred, alpha_pred, cv::Size(w, h));
    // need clone to allocate a new continuous memory if not performed resize.
    // The memory elements point to will release after return.
  else alpha_pred = alpha_pred.clone();

  cv::Mat pmat = alpha_pred; // ref
  content.pha_mat = pmat; // auto handle the memory inside ocv with smart ref.

  if (!minimum_post_process)
  {
    // MODNet only predict Alpha, no fgr. So,
    // the fake fgr and merge mat may not need,
    // let the fgr mat and merge mat empty to
    // speed up the post processes.
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

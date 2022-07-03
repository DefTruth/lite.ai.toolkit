//
// Created by DefTruth on 2022/4/9.
//

#include "mnn_backgroundmattingv2.h"
#include "lite/utils.h"

using mnncv::MNNBackgroundMattingV2;

MNNBackgroundMattingV2::MNNBackgroundMattingV2(
    const std::string &_mnn_path,
    unsigned int _num_threads
) : log_id(_mnn_path.data()),
    mnn_path(_mnn_path.data()),
    num_threads(_num_threads)
{
  initialize_interpreter();
  initialize_pretreat();
}

MNNBackgroundMattingV2::~MNNBackgroundMattingV2()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNBackgroundMattingV2::initialize_interpreter()
{
  // 1. init interpreter
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  // 2. init schedule_config
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  // 3. create session
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // 4. init input tensor
  src_tensor = mnn_interpreter->getSessionInput(mnn_session, "src");
  bgr_tensor = mnn_interpreter->getSessionInput(mnn_session, "bgr");
  // 5. init input dims
  input_height = src_tensor->height();
  input_width = src_tensor->width();
  dimension_type = src_tensor->getDimensionType(); // CAFFE
  mnn_interpreter->resizeTensor(src_tensor, src_tensor->shape());
  mnn_interpreter->resizeTensor(bgr_tensor, bgr_tensor->shape());
  mnn_interpreter->resizeSession(mnn_session);
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

void MNNBackgroundMattingV2::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (src_tensor) src_tensor->printShape();
  if (bgr_tensor) bgr_tensor->printShape();
  if (dimension_type == MNN::Tensor::CAFFE)
    std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
  else if (dimension_type == MNN::Tensor::TENSORFLOW)
    std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
  else if (dimension_type == MNN::Tensor::CAFFE_C4)
    std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  auto tmp_output_map = mnn_interpreter->getSessionOutputAll(mnn_session);
  std::cout << "getSessionOutputAll done!\n";
  for (auto it = tmp_output_map.cbegin(); it != tmp_output_map.cend(); ++it)
  {
    std::cout << "Output: " << it->first << ": ";
    it->second->printShape();
  }
  std::cout << "========================================\n";
}

void MNNBackgroundMattingV2::initialize_pretreat()
{
  pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
      MNN::CV::ImageProcess::create(
          MNN::CV::BGR,
          MNN::CV::RGB,
          mean_vals, 3,
          norm_vals, 3
      )
  );
}

inline void MNNBackgroundMattingV2::transform(const cv::Mat &mat, const cv::Mat &bgr)
{
  cv::Mat mat_rs, bgr_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::resize(bgr, bgr_rs, cv::Size(input_width, input_height));
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], src_tensor);
  pretreat->convert(bgr_rs.data, input_width, input_height, bgr_rs.step[0], bgr_tensor);
}

void MNNBackgroundMattingV2::detect(const cv::Mat &mat, const cv::Mat &bgr,
                                    types::MattingContent &content, bool remove_noise,
                                    bool minimum_post_process)
{
  if (mat.empty() || bgr.empty()) return;
  // 1. make input tensor
  this->transform(mat, bgr);
  // 2. inference & run session
  mnn_interpreter->runSession(mnn_session);

  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise, minimum_post_process);
}

void MNNBackgroundMattingV2::generate_matting(
    const std::map<std::string, MNN::Tensor *> &output_tensors, const cv::Mat &mat,
    types::MattingContent &content, bool remove_noise,
    bool minimum_post_process)
{
  auto device_fgr_ptr = output_tensors.at("fgr");
  auto device_pha_ptr = output_tensors.at("pha");
  MNN::Tensor host_fgr_tensor(device_fgr_ptr, device_fgr_ptr->getDimensionType());  // NCHW
  MNN::Tensor host_pha_tensor(device_pha_ptr, device_pha_ptr->getDimensionType());  // NCHW
  device_fgr_ptr->copyToHostTensor(&host_fgr_tensor);
  device_pha_ptr->copyToHostTensor(&host_pha_tensor);
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;
  const unsigned int out_h = input_height;
  const unsigned int out_w = input_width;

  float *fgr_ptr = host_fgr_tensor.host<float>();
  float *pha_ptr = host_pha_tensor.host<float>();
  const unsigned int channel_step = out_h * out_w;

  // fast assign & channel transpose(CHW->HWC).
  cv::Mat pmat(out_h, out_w, CV_32FC1, pha_ptr);
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  std::vector<cv::Mat> fgr_channel_mats;
  cv::Mat rmat(out_h, out_w, CV_32FC1, fgr_ptr);
  cv::Mat gmat(out_h, out_w, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat(out_h, out_w, CV_32FC1, fgr_ptr + 2 * channel_step);
  rmat *= 255.;
  bmat *= 255.;
  gmat *= 255.;
  fgr_channel_mats.push_back(bmat);
  fgr_channel_mats.push_back(gmat);
  fgr_channel_mats.push_back(rmat);

  content.pha_mat = pmat;
  cv::merge(fgr_channel_mats, content.fgr_mat);
  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);

  if (!minimum_post_process)
  {
    std::vector<cv::Mat> merge_channel_mats;
    cv::Mat rest = 1. - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.;
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);
    cv::merge(merge_channel_mats, content.merge_mat);
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  // resize alpha
  if (out_h != h || out_w != w)
  {
    // already allocated a new continuous memory after resize (pha_mat)
    cv::resize(content.pha_mat, content.pha_mat, cv::Size(w, h));
    cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(w, h));
    if (!minimum_post_process)
      cv::resize(content.merge_mat, content.merge_mat, cv::Size(w, h));
  } //
  else
  {
    // need clone to allocate a new continuous memory if not performed resize.
    // The memory elements point to will release after return.
    content.pha_mat = content.pha_mat.clone();
  }

  content.flag = true;
}



































//
// Created by DefTruth on 2022/4/9.
//

#include "tnn_backgroundmattingv2.h"
#include "lite/utils.h"

using tnncv::TNNBackgroundMattingV2;

TNNBackgroundMattingV2::TNNBackgroundMattingV2(
    const std::string &_proto_path,
    const std::string &_model_path,
    unsigned int _num_threads
) : proto_path(_proto_path.data()),
    model_path(_model_path.data()),
    log_id(_proto_path.data()),
    num_threads(_num_threads)
{
  initialize_instance();
}

TNNBackgroundMattingV2::~TNNBackgroundMattingV2()
{
  net = nullptr;
  src_mat = nullptr;
  bgr_mat = nullptr;
  instance = nullptr;
}

void TNNBackgroundMattingV2::initialize_instance()
{
  std::string proto_content_buffer, model_content_buffer;
  proto_content_buffer = BasicTNNHandler::content_buffer_from(proto_path);
  model_content_buffer = BasicTNNHandler::content_buffer_from(model_path);

  tnn::ModelConfig model_config;
  model_config.model_type = tnn::MODEL_TYPE_TNN;
  model_config.params = {proto_content_buffer, model_content_buffer};

  // 1. init TNN net
  tnn::Status status;
  net = std::make_shared<tnn::TNN>();
  status = net->Init(model_config);
  if (status != tnn::TNN_OK || !net)
  {
#ifdef LITETNN_DEBUG
    std::cout << "net->Init failed!\n";
#endif
    return;
  }
  // 2. init device type, change this default setting
  // for better performance. such as CUDA/OPENCL/...
#ifdef __ANDROID__
  network_device_type = tnn::DEVICE_ARM; // CPU,GPU
  input_device_type = tnn::DEVICE_ARM; // CPU only
  output_device_type = tnn::DEVICE_ARM;
#else
  network_device_type = tnn::DEVICE_X86; // CPU,GPU
  input_device_type = tnn::DEVICE_X86; // CPU only
  output_device_type = tnn::DEVICE_X86;
#endif
  // 3. init instance
  tnn::NetworkConfig network_config;
  network_config.library_path = {""};
  network_config.device_type = network_device_type;

  instance = net->CreateInst(network_config, status);
  if (status != tnn::TNN_OK || !instance)
  {
#ifdef LITETNN_DEBUG
    std::cout << "CreateInst failed!" << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. setting up num_threads
  instance->SetCpuNumThreads((int) num_threads);
  // 5. init input information.
  for (auto &name: input_names)
    input_shapes[name] = BasicTNNHandler::get_input_shape(instance, name);
  auto src_shape = input_shapes.at("src");
  if (src_shape.size() != 4)
  {
#ifdef LITETNN_DEBUG
    throw std::runtime_error("Found src_shape.size()!=4, but "
                             "src input only support 4 dims."
                             "Such as NCHW, NHWC ...");
#else
    return;
#endif
  }
  input_mat_type = BasicTNNHandler::get_input_mat_type(instance, "src");
  input_data_format = BasicTNNHandler::get_input_data_format(instance, "src");
  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    input_height = src_shape.at(2);
    input_width = src_shape.at(3);
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    input_height = src_shape.at(1);
    input_width = src_shape.at(2);
  } // unsupport
  else
  {
#ifdef LITETNN_DEBUG
    std::cout << "src input only support NCHW and NHWC "
                 "input_data_format, but found others.\n";
#endif
    return;
  }
  // 6. init output information, debug only.
  for (auto &name: output_names)
    output_shapes[name] = BasicTNNHandler::get_output_shape(instance, name);
#ifdef LITETNN_DEBUG
  this->print_debug_string();
#endif
}

void TNNBackgroundMattingV2::print_debug_string()
{
  std::cout << "LITETNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  for (auto &in: input_shapes)
    BasicTNNHandler::print_name_shape(in.first, in.second);
  std::string data_format_string =
      (input_data_format == tnn::DATA_FORMAT_NCHW) ? "NCHW" : "NHWC";
  std::cout << "Input Data Format: " << data_format_string << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (auto &out: output_shapes)
    BasicTNNHandler::print_name_shape(out.first, out.second);
  std::cout << "========================================\n";
}

void TNNBackgroundMattingV2::transform(const cv::Mat &mat_rs, const cv::Mat &bgr_rs)
{
  // push into src_mat
  src_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::N8UC3,
      input_shapes.at("src"),
      (void *) mat_rs.data
  );
  if (!src_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "src_mat == nullptr! transform failed\n";
#endif
  }
  bgr_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::N8UC3,
      input_shapes.at("bgr"),
      (void *) bgr_rs.data
  );
  if (!bgr_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "bgr_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNBackgroundMattingV2::detect(const cv::Mat &mat, const cv::Mat &bgr,
                                    types::MattingContent &content, bool remove_noise,
                                    bool minimum_post_process)
{
  if (mat.empty() || bgr.empty()) return;
  cv::Mat mat_rs, bgr_rs;
  // resize mat outside 'transform' to prevent memory overflow
  // reference: https://github.com/DefTruth/lite.ai.toolkit/issues/240
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::resize(bgr, bgr_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  cv::cvtColor(bgr_rs, bgr_rs, cv::COLOR_BGR2RGB);
  this->transform(mat_rs, bgr_rs);
  // 2. set input_mat
  tnn::MatConvertParam cvt_param;
  cvt_param.scale = scale_vals;
  cvt_param.bias = bias_vals;

  auto status_src = instance->SetInputMat(src_mat, cvt_param, "src");
  auto status_bgr = instance->SetInputMat(bgr_mat, cvt_param, "bgr");
  if (status_src != tnn::TNN_OK || status_bgr != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->SetInputMat failed!:"
              << status_src.description().c_str() << ": "
              << status_bgr.description().c_str() << "\n";
#endif
    return;
  }
// 3. forward
  auto status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->Forward failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. generate matting
  this->generate_matting(instance, mat, content, remove_noise, minimum_post_process);
}

void TNNBackgroundMattingV2::generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                                              const cv::Mat &mat, types::MattingContent &content,
                                              bool remove_noise, bool minimum_post_process)
{
  std::shared_ptr<tnn::Mat> fgr_mat;
  std::shared_ptr<tnn::Mat> pha_mat;
  tnn::MatConvertParam cvt_param;
  tnn::Status status_fgr, status_pha;

  status_fgr = _instance->GetOutputMat(fgr_mat, cvt_param, "fgr", output_device_type);
  status_pha = _instance->GetOutputMat(pha_mat, cvt_param, "pha", output_device_type);

  if (status_fgr != tnn::TNN_OK || status_pha != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status_fgr.description().c_str() << ": "
              << status_pha.description().c_str() << "\n";
#endif
    return;
  }
  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;
  const unsigned int out_h = input_height;
  const unsigned int out_w = input_width;

  float *fgr_ptr = (float *) fgr_mat->GetData();
  float *pha_ptr = (float *) pha_mat->GetData();
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
    cv::resize(content.pha_mat, content.pha_mat, cv::Size(w, h));
    cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(w, h));
    if (!minimum_post_process)
      cv::resize(content.merge_mat, content.merge_mat, cv::Size(w, h));
  }

  content.flag = true;
}


















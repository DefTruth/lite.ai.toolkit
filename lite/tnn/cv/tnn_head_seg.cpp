//
// Created by DefTruth on 2022/6/11.
//

#include "tnn_head_seg.h"

using tnncv::TNNHeadSeg;

TNNHeadSeg::TNNHeadSeg(
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

TNNHeadSeg::~TNNHeadSeg()
{
  net = nullptr;
  input_mat = nullptr;
  instance = nullptr;
}

void TNNHeadSeg::initialize_instance()
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
  network_config.data_format = tnn::DATA_FORMAT_NHWC;

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
  input_shape = BasicTNNHandler::get_input_shape(instance, "input_1_0");
  // hard code (NHWC) from pb -> ONNX -> TNN
  input_batch = input_shape.at(0);
  input_height = input_shape.at(1);
  input_width = input_shape.at(2);
  input_channel = input_shape.at(3);

  if (input_shape.size() != 4)
  {
#ifdef LITETNN_DEBUG
    throw std::runtime_error("Found input_shape.size()!=4, but "
                             "input only support 4 dims."
                             "Such as NCHW, NHWC ...");
#else
    return;
#endif
  }
  input_mat_type = BasicTNNHandler::get_input_mat_type(instance, "input_1_0");
  input_data_format = BasicTNNHandler::get_input_data_format(instance, "input_1_0");
  // 6. init output information, debug only.
  output_shape = BasicTNNHandler::get_output_shape(instance, "sigmoid/Sigmoid:0");
#ifdef LITETNN_DEBUG
  this->print_debug_string();
#endif
}

void TNNHeadSeg::print_debug_string()
{
  std::cout << "LITETNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  BasicTNNHandler::print_name_shape("input_1_0", input_shape);
  std::string data_format_string =
      (input_data_format == tnn::DATA_FORMAT_NCHW) ? "NCHW" : "NHWC";
  std::cout << "Input Data Format: " << data_format_string << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  BasicTNNHandler::print_name_shape("sigmoid/Sigmoid:0", output_shape);
  std::cout << "========================================\n";
}

void TNNHeadSeg::transform(const cv::Mat &mat_rs)
{
  //  be carefully, no deepcopy inside this tnn::Mat constructor,
  //  so, we can not pass a local cv::Mat to this constructor.
  //  push into input_mat
  input_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::N8UC3,
      input_shape,
      (void *) mat_rs.data
  );
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNHeadSeg::detect(const cv::Mat &mat, types::HeadSegContent &content)
{
  if (mat.empty()) return;
  const unsigned int img_h = mat.rows;
  const unsigned int img_w = mat.cols;
  const unsigned int channels = mat.channels();
  if (channels != 3) return;
  const unsigned int input_h = input_height; // 384
  const unsigned int input_w = input_width; // 384

  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_w, input_h));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  // 1. make input tensor
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
  // 4. fetch
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> mask_pred; // (1,384,384,1)
  status = instance->GetOutputMat(mask_pred, cvt_param, "sigmoid/Sigmoid:0", output_device_type);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto mask_dims = mask_pred->GetDims();
#ifdef LITETNN_DEBUG
  BasicTNNHandler::print_name_shape("sigmoid/Sigmoid:0", mask_dims);
#endif

  const unsigned int out_h = mask_dims.at(1);
  const unsigned int out_w = mask_dims.at(2);
  float *mask_ptr = (float *) mask_pred->GetData();

  cv::Mat mask_adj;
  cv::Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr);
  cv::resize(mask_out, mask_adj, cv::Size(img_w, img_h)); // (img_h,img_w,1)

  content.mask = mask_adj;
  content.flag = true;
}



















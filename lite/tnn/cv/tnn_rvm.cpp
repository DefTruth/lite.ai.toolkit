//
// Created by DefTruth on 2021/10/18.
//

#include "tnn_rvm.h"
#include "lite/utils.h"


using tnncv::TNNRobustVideoMatting;

TNNRobustVideoMatting::TNNRobustVideoMatting(
    const std::string &_proto_path,
    const std::string &_model_path,
    unsigned int _num_threads
) : proto_path(_proto_path.data()),
    model_path(_model_path.data()),
    log_id(_proto_path.data()),
    num_threads(_num_threads)
{
  initialize_instance();
  initialize_context();
}

TNNRobustVideoMatting::~TNNRobustVideoMatting()
{
  net = nullptr;
  src_mat = nullptr;
  r1i_mat = nullptr;
  r2i_mat = nullptr;
  r3i_mat = nullptr;
  r4i_mat = nullptr;
  instance = nullptr;
}

void TNNRobustVideoMatting::initialize_instance()
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
  src_size = 1 * 3 * input_height * input_width;
  // 6. init output information, debug only.
  for (auto &name: output_names)
    output_shapes[name] = BasicTNNHandler::get_output_shape(instance, name);
#ifdef LITETNN_DEBUG
  this->print_debug_string();
#endif
}

int TNNRobustVideoMatting::value_size_of(tnn::DimsVector &shape)
{
  if (shape.empty()) return 0;
  int _size = 1;
  for (auto &s: shape) _size *= s;
  return _size;
}

void TNNRobustVideoMatting::print_debug_string()
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

void TNNRobustVideoMatting::initialize_context()
{
  r1i_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::NCHW_FLOAT,
      input_shapes.at("r1i")
  );
  r2i_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::NCHW_FLOAT,
      input_shapes.at("r2i")
  );
  r3i_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::NCHW_FLOAT,
      input_shapes.at("r3i")
  );
  r4i_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::NCHW_FLOAT,
      input_shapes.at("r4i")
  );
  r1i_size = this->value_size_of(input_shapes.at("r1i"));
  r2i_size = this->value_size_of(input_shapes.at("r2i"));
  r3i_size = this->value_size_of(input_shapes.at("r3i"));
  r4i_size = this->value_size_of(input_shapes.at("r4i"));
  // init 0.
  std::fill_n((float *) r1i_mat->GetData(), r1i_size, 0.f);
  std::fill_n((float *) r2i_mat->GetData(), r2i_size, 0.f);
  std::fill_n((float *) r3i_mat->GetData(), r3i_size, 0.f);
  std::fill_n((float *) r4i_mat->GetData(), r4i_size, 0.f);

  context_is_initialized = true;
}

void TNNRobustVideoMatting::transform(const cv::Mat &mat_rs)
{
  //  cv::Mat canvas;
  //  cv::resize(mat, canvas, cv::Size(input_width, input_height));
  //  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  //  reference: https://github.com/DefTruth/lite.ai.toolkit/issues/240
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
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNRobustVideoMatting::detect(const cv::Mat &mat, types::MattingContent &content, bool video_mode,
                                   bool remove_noise, bool minimum_post_process)
{
  if (mat.empty()) return;
  int img_h = mat.rows;
  int img_w = mat.cols;
  if (!context_is_initialized) return;

  // 1. make input tensor
  cv::Mat mat_rs;
  // resize mat outside 'transform' to prevent memory overflow
  // reference: https://github.com/DefTruth/lite.ai.toolkit/issues/240
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  this->transform(mat_rs);
  // 2. set input_mat
  tnn::MatConvertParam src_cvt_param, ctx_cvt_param;
  src_cvt_param.scale = scale_vals;
  src_cvt_param.bias = bias_vals;

  tnn::Status status_src, status_r1i, status_r2i, status_r3i, status_r4i;
  status_src = instance->SetInputMat(src_mat, src_cvt_param, "src");
  status_r1i = instance->SetInputMat(r1i_mat, ctx_cvt_param, "r1i");
  status_r2i = instance->SetInputMat(r2i_mat, ctx_cvt_param, "r2i");
  status_r3i = instance->SetInputMat(r3i_mat, ctx_cvt_param, "r3i");
  status_r4i = instance->SetInputMat(r4i_mat, ctx_cvt_param, "r4i");
  if (status_src != tnn::TNN_OK || status_r1i != tnn::TNN_OK ||
      status_r2i != tnn::TNN_OK || status_r3i != tnn::TNN_OK ||
      status_r4i != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->SetInputMat failed!:"
              << status_src.description().c_str() << ": "
              << status_r1i.description().c_str() << ": "
              << status_r2i.description().c_str() << ": "
              << status_r3i.description().c_str() << ": "
              << status_r4i.description().c_str() << "\n";
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
  this->generate_matting(instance, content, img_h, img_w, remove_noise, minimum_post_process);
  // 5. update context (needed for video matting)
  if (video_mode)
  {
    context_is_update = false; // init state.
    this->update_context(instance);
  }

}

void TNNRobustVideoMatting::detect_video(
    const std::string &video_path, const std::string &output_path,
    std::vector<types::MattingContent> &contents, bool save_contents,
    unsigned int writer_fps, bool remove_noise, bool minimum_post_process,
    const cv::Mat &background)
{
// 0. init video capture
  cv::VideoCapture video_capture(video_path);
  const unsigned int width = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
  const unsigned int height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  const unsigned int frame_count = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
  if (!video_capture.isOpened())
  {
    std::cout << "Can not open video: " << video_path << "\n";
    return;
  }
  // 1. init video writer
  cv::VideoWriter video_writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               writer_fps, cv::Size(width, height));
  if (!video_writer.isOpened())
  {
    std::cout << "Can not open writer: " << output_path << "\n";
    return;
  }

  // 2. matting loop
  cv::Mat mat;
  unsigned int i = 0;
  while (video_capture.read(mat))
  {
    i += 1;
    types::MattingContent content;
    this->detect(mat, content, true, remove_noise, minimum_post_process); // video_mode true
    // 3. save contents and writing out.
    if (content.flag)
    {
//      if (save_contents) contents.push_back(content);
//      if (!content.merge_mat.empty()) video_writer.write(content.merge_mat);

      if (save_contents) contents.push_back(content);
      // 3.1 do nothing if set minimum_post_process as true
      if (background.empty())
      {
        if (!content.merge_mat.empty() && !minimum_post_process)
          video_writer.write(content.merge_mat);
        else if (!content.fgr_mat.empty())
          video_writer.write(content.fgr_mat);
      } //
      else
      {
        cv::Mat out_mat;
        // 3.2 merge user custom background
        if (!content.pha_mat.empty())
        {
          if (!content.fgr_mat.empty())
            lite::utils::swap_background(content.fgr_mat, content.pha_mat,
                                         background, out_mat, false);
          else
            lite::utils::swap_background(mat, content.pha_mat,
                                         background, out_mat, false);
        }
        if (!out_mat.empty()) video_writer.write(out_mat);

      }

    }
    // 4. check context states.
    if (!context_is_update) break;
#ifdef LITETNN_DEBUG
    std::cout << i << "/" << frame_count << " done!" << "\n";
#endif
  }

  // 5. release
  video_capture.release();
  video_writer.release();
}

void TNNRobustVideoMatting::generate_matting(std::shared_ptr<tnn::Instance> &_instance,
                                             types::MattingContent &content,
                                             int img_h, int img_w,
                                             bool remove_noise,
                                             bool minimum_post_process)
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

  float *fgr_ptr = (float *) fgr_mat->GetData();
  float *pha_ptr = (float *) pha_mat->GetData();
  const unsigned int channel_step = input_height * input_width;

  // fast assign & channel transpose(CHW->HWC).
  cv::Mat rmat(input_height, input_width, CV_32FC1, fgr_ptr);
  cv::Mat gmat(input_height, input_width, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat(input_height, input_width, CV_32FC1, fgr_ptr + 2 * channel_step);
  cv::Mat pmat(input_height, input_width, CV_32FC1, pha_ptr); // ref only, zero-copy.
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  rmat *= 255.f;
  bmat *= 255.f;
  gmat *= 255.f;
  std::vector<cv::Mat> fgr_channel_mats;
  fgr_channel_mats.push_back(bmat);
  fgr_channel_mats.push_back(gmat);
  fgr_channel_mats.push_back(rmat);

  // need clone to allocate a new continuous memory.
  content.pha_mat = pmat.clone(); // allocated
  cv::merge(fgr_channel_mats, content.fgr_mat);
  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);

  if (!minimum_post_process)
  {
    cv::Mat rest = 1.f - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.f;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.f;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.f;
    std::vector<cv::Mat> merge_channel_mats;
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);
    cv::merge(merge_channel_mats, content.merge_mat);
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  if (img_w != input_width || img_h != input_height)
  {
    cv::resize(content.pha_mat, content.pha_mat, cv::Size(img_w, img_h));
    cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(img_w, img_h));
    if (!minimum_post_process)
      cv::resize(content.merge_mat, content.merge_mat, cv::Size(img_w, img_h));
  }

  content.flag = true;
}

void TNNRobustVideoMatting::update_context(std::shared_ptr<tnn::Instance> &_instance)
{
  std::shared_ptr<tnn::Mat> r1o_mat;
  std::shared_ptr<tnn::Mat> r2o_mat;
  std::shared_ptr<tnn::Mat> r3o_mat;
  std::shared_ptr<tnn::Mat> r4o_mat;
  tnn::MatConvertParam cvt_param;
  tnn::Status status_r1o;
  tnn::Status status_r2o;
  tnn::Status status_r3o;
  tnn::Status status_r4o;

  status_r1o = _instance->GetOutputMat(r1o_mat, cvt_param, "r1o", output_device_type);
  status_r2o = _instance->GetOutputMat(r2o_mat, cvt_param, "r2o", output_device_type);
  status_r3o = _instance->GetOutputMat(r3o_mat, cvt_param, "r3o", output_device_type);
  status_r4o = _instance->GetOutputMat(r4o_mat, cvt_param, "r4o", output_device_type);

  if (status_r1o != tnn::TNN_OK || status_r2o != tnn::TNN_OK ||
      status_r3o != tnn::TNN_OK || status_r4o != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat context failed!:"
              << status_r1o.description().c_str() << ": "
              << status_r2o.description().c_str() << ": "
              << status_r3o.description().c_str() << ": "
              << status_r4o.description().c_str() << "\n";
#endif
    return;
  }
  void *command_queue = nullptr;
  auto status_cmd = _instance->GetCommandQueue(&command_queue);
  if (status_cmd != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetCommandQueue failed!:"
              << status_cmd.description().c_str() << "\n";
#endif
    return;
  }

  tnn::MatUtils::Copy(*r1o_mat, *r1i_mat, command_queue);
  tnn::MatUtils::Copy(*r2o_mat, *r2i_mat, command_queue);
  tnn::MatUtils::Copy(*r3o_mat, *r3i_mat, command_queue);
  tnn::MatUtils::Copy(*r4o_mat, *r4i_mat, command_queue);

  context_is_update = true;
}































//
// Created by DefTruth on 2021/10/10.
//

#include "mnn_rvm.h"

using mnncv::MNNRobustVideoMatting;

MNNRobustVideoMatting::MNNRobustVideoMatting(
    const std::string &_mnn_path,
    unsigned int _num_threads,
    unsigned int _variant_type
) : log_id(_mnn_path.data()),
    mnn_path(_mnn_path.data()),
    num_threads(_num_threads),
    variant_type(_variant_type)
{
  initialize_interpreter();
  initialize_context();
  initialize_pretreat();
}

MNNRobustVideoMatting::~MNNRobustVideoMatting()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNRobustVideoMatting::initialize_interpreter()
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
  // 5. init input dims
  input_height = src_tensor->height();
  input_width = src_tensor->width();
  dimension_type = src_tensor->getDimensionType(); // CAFFE
  mnn_interpreter->resizeTensor(src_tensor, {1, 3, input_height, input_width});
  mnn_interpreter->resizeSession(mnn_session);
  src_size = 1 * 3 * input_height * input_width;
  // 6. rxi
  r1i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r1i");
  r2i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r2i");
  r3i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r3i");
  r4i_tensor = mnn_interpreter->getSessionInput(mnn_session, "r4i");
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

void MNNRobustVideoMatting::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (src_tensor) src_tensor->printShape();
  if (r1i_tensor) r1i_tensor->printShape();
  if (r2i_tensor) r2i_tensor->printShape();
  if (r3i_tensor) r3i_tensor->printShape();
  if (r4i_tensor) r4i_tensor->printShape();
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

void MNNRobustVideoMatting::initialize_context()
{
  if (variant_type == VARIANT::MOBILENETV3)
  {
    if (input_width == 1920 && input_height == 1080)
    {
      mnn_interpreter->resizeTensor(r1i_tensor, {1, 16, 135, 240});
      mnn_interpreter->resizeTensor(r2i_tensor, {1, 20, 68, 120});
      mnn_interpreter->resizeTensor(r3i_tensor, {1, 40, 34, 60});
      mnn_interpreter->resizeTensor(r4i_tensor, {1, 64, 17, 30});
      r1i_size = 1 * 16 * 135 * 240;
      r2i_size = 1 * 20 * 68 * 120;
      r3i_size = 1 * 40 * 34 * 60;
      r4i_size = 1 * 64 * 17 * 30;
    } // hxw 480x640 480x480 640x480
    else
    {
      mnn_interpreter->resizeTensor(r1i_tensor, {1, 16, input_height / 2, input_width / 2});
      mnn_interpreter->resizeTensor(r2i_tensor, {1, 20, input_height / 4, input_width / 4});
      mnn_interpreter->resizeTensor(r3i_tensor, {1, 40, input_height / 8, input_width / 8});
      mnn_interpreter->resizeTensor(r4i_tensor, {1, 64, input_height / 16, input_width / 16});
      r1i_size = 1 * 16 * (input_height / 2) * (input_width / 2);
      r2i_size = 1 * 20 * (input_height / 4) * (input_width / 4);
      r3i_size = 1 * 40 * (input_height / 8) * (input_width / 8);
      r4i_size = 1 * 64 * (input_height / 16) * (input_width / 16);
    }
  }// RESNET50
  else
  {
    if (input_width == 1920 && input_height == 1080)
    {
      mnn_interpreter->resizeTensor(r1i_tensor, {1, 16, 135, 240});
      mnn_interpreter->resizeTensor(r2i_tensor, {1, 32, 68, 120});
      mnn_interpreter->resizeTensor(r3i_tensor, {1, 64, 34, 60});
      mnn_interpreter->resizeTensor(r4i_tensor, {1, 128, 17, 30});
      r1i_size = 1 * 16 * 135 * 240;
      r2i_size = 1 * 32 * 68 * 120;
      r3i_size = 1 * 64 * 34 * 60;
      r4i_size = 1 * 128 * 17 * 30;
    } // hxw 480x640 480x480 640x480
    else
    {
      mnn_interpreter->resizeTensor(r1i_tensor, {1, 16, input_height / 2, input_width / 2});
      mnn_interpreter->resizeTensor(r2i_tensor, {1, 32, input_height / 4, input_width / 4});
      mnn_interpreter->resizeTensor(r3i_tensor, {1, 64, input_height / 8, input_width / 8});
      mnn_interpreter->resizeTensor(r4i_tensor, {1, 128, input_height / 16, input_width / 16});
      r1i_size = 1 * 16 * (input_height / 2) * (input_width / 2);
      r2i_size = 1 * 32 * (input_height / 4) * (input_width / 4);
      r3i_size = 1 * 64 * (input_height / 8) * (input_width / 8);
      r4i_size = 1 * 128 * (input_height / 16) * (input_width / 16);
    }
  }
  // resize session
  mnn_interpreter->resizeSession(mnn_session);
  // init 0.
  std::fill_n(r1i_tensor->host<float>(), r1i_size, 0.f);
  std::fill_n(r2i_tensor->host<float>(), r2i_size, 0.f);
  std::fill_n(r3i_tensor->host<float>(), r3i_size, 0.f);
  std::fill_n(r4i_tensor->host<float>(), r4i_size, 0.f);

  context_is_initialized = true;
}

inline void MNNRobustVideoMatting::initialize_pretreat()
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

inline void MNNRobustVideoMatting::transform(const cv::Mat &mat_rs)
{
  pretreat->convert(mat_rs.data, input_width, input_height, mat_rs.step[0], src_tensor);
}

void MNNRobustVideoMatting::detect(const cv::Mat &mat, types::MattingContent &content)
{
  if (mat.empty()) return;
  int img_h = mat.rows;
  int img_w = mat.cols;
  if (!context_is_initialized) return;

  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // 1. make input tensor
  this->transform(mat_rs);

  // 2. inference & run session
  mnn_interpreter->runSession(mnn_session);

  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate matting
  this->generate_matting(output_tensors, content, img_h, img_w);
  // 4.  update context (needed for video matting)
  context_is_update = false; // init state.
  this->update_context(output_tensors);
}

void MNNRobustVideoMatting::detect_video(
    const std::string &video_path, const std::string &output_path,
    std::vector<types::MattingContent> &contents, bool save_contents,
    unsigned int writer_fps)
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
    this->detect(mat, content);
    // 3. save contents and writing out.
    if (content.flag)
    {
      if (save_contents) contents.push_back(content);
      if (!content.merge_mat.empty()) video_writer.write(content.merge_mat);
    }
    // 4. check context states.
    if (!context_is_update) break;
#ifdef LITEMNN_DEBUG
    std::cout << i << "/" << frame_count << " done!" << "\n";
#endif
  }

  // 5. release
  video_capture.release();
  video_writer.release();
}

void MNNRobustVideoMatting::generate_matting(
    const std::map<std::string, MNN::Tensor *> &output_tensors,
    types::MattingContent &content,
    int img_h, int img_w)
{
  auto device_fgr_ptr = output_tensors.at("fgr");
  auto device_pha_ptr = output_tensors.at("pha");
  MNN::Tensor host_fgr_tensor(device_fgr_ptr, device_fgr_ptr->getDimensionType());  // NCHW
  MNN::Tensor host_pha_tensor(device_pha_ptr, device_pha_ptr->getDimensionType());  // NCHW
  device_fgr_ptr->copyToHostTensor(&host_fgr_tensor);
  device_pha_ptr->copyToHostTensor(&host_pha_tensor);

  float *fgr_ptr = host_fgr_tensor.host<float>();
  float *pha_ptr = host_pha_tensor.host<float>();
  const unsigned int channel_step = input_height * input_width;

  // fast assign & channel transpose(CHW->HWC).
  cv::Mat rmat(input_height, input_width, CV_32FC1, fgr_ptr);
  cv::Mat gmat(input_height, input_width, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat(input_height, input_width, CV_32FC1, fgr_ptr + 2 * channel_step);
  cv::Mat pmat(input_height, input_width, CV_32FC1, pha_ptr); // ref only, zero-copy.
  rmat *= 255.f;
  bmat *= 255.f;
  gmat *= 255.f;
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

  if (img_w != input_width || img_h != input_height)
  {
    cv::resize(content.pha_mat, content.pha_mat, cv::Size(img_w, img_h));
    cv::resize(content.fgr_mat, content.fgr_mat, cv::Size(img_w, img_h));
    cv::resize(content.merge_mat, content.merge_mat, cv::Size(img_w, img_h));
  }

  content.flag = true;
}

void MNNRobustVideoMatting::update_context(const std::map<std::string, MNN::Tensor *> &output_tensors)
{
  auto device_r1o_ptr = output_tensors.at("r1o");
  auto device_r2o_ptr = output_tensors.at("r2o");
  auto device_r3o_ptr = output_tensors.at("r3o");
  auto device_r4o_ptr = output_tensors.at("r4o");

  device_r1o_ptr->copyToHostTensor(r1i_tensor);
  device_r2o_ptr->copyToHostTensor(r2i_tensor);
  device_r3o_ptr->copyToHostTensor(r3i_tensor);
  device_r4o_ptr->copyToHostTensor(r4i_tensor);

  context_is_update = true;
}

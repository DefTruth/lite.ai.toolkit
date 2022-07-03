//
// Created by DefTruth on 2021/9/20.
//

#include "rvm.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::RobustVideoMatting;

RobustVideoMatting::RobustVideoMatting(const std::string &_onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
#ifdef LITE_WIN32
  std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
  onnx_path = _w_onnx_path.data();
#else
  onnx_path = _onnx_path.data();
#endif
  ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);
  // 0. session options
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(num_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(4);
  // 1. session
  // GPU Compatibility.
#ifdef USE_CUDA
  OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); // C API stable.
#endif
  ort_session = new Ort::Session(ort_env, onnx_path, session_options);
#if LITEORT_DEBUG
  std::cout << "Load " << onnx_path << " done!" << std::endl;
#endif
}

RobustVideoMatting::~RobustVideoMatting()
{
  if (ort_session)
    delete ort_session;
  ort_session = nullptr;
}

int64_t RobustVideoMatting::value_size_of(const std::vector<int64_t> &dims)
{
  if (dims.empty()) return 0;
  int64_t value_size = 1;
  for (const auto &size: dims) value_size *= size;
  return value_size;
}

std::vector<Ort::Value> RobustVideoMatting::transform(const cv::Mat &mat)
{
  cv::Mat src = mat.clone();
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  std::vector<int64_t> &src_dims = dynamic_input_node_dims.at(0); // (1,3,h,w)
  // update src height and width
  src_dims.at(2) = img_height;
  src_dims.at(3) = img_width;
  // assume that rxi's dims and value_handler was updated by last step in a while loop.
  std::vector<int64_t> &r1i_dims = dynamic_input_node_dims.at(1); // (1,?,?h,?w)
  std::vector<int64_t> &r2i_dims = dynamic_input_node_dims.at(2); // (1,?,?h,?w)
  std::vector<int64_t> &r3i_dims = dynamic_input_node_dims.at(3); // (1,?,?h,?w)
  std::vector<int64_t> &r4i_dims = dynamic_input_node_dims.at(4); // (1,?,?h,?w)
  std::vector<int64_t> &dsr_dims = dynamic_input_node_dims.at(5); // (1)
  int64_t src_value_size = this->value_size_of(src_dims); // (1*3*h*w)
  int64_t r1i_value_size = this->value_size_of(r1i_dims); // (1*?*?h*?w)
  int64_t r2i_value_size = this->value_size_of(r2i_dims); // (1*?*?h*?w)
  int64_t r3i_value_size = this->value_size_of(r3i_dims); // (1*?*?h*?w)
  int64_t r4i_value_size = this->value_size_of(r4i_dims); // (1*?*?h*?w)
  int64_t dsr_value_size = this->value_size_of(dsr_dims); // 1
  dynamic_src_value_handler.resize(src_value_size);

  // normalize & RGB
  cv::cvtColor(src, src, cv::COLOR_BGR2RGB); // (h,w,3)
  src.convertTo(src, CV_32FC3, 1.0f / 255.0f, 0.f); // 0.~1.

  // convert to tensor.
  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(ortcv::utils::transform::create_tensor(
      src, src_dims, memory_info_handler, dynamic_src_value_handler,
      ortcv::utils::transform::CHW
  ));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, dynamic_r1i_value_handler.data(),
      r1i_value_size, r1i_dims.data(), r1i_dims.size()
  ));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, dynamic_r2i_value_handler.data(),
      r2i_value_size, r2i_dims.data(), r2i_dims.size()
  ));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, dynamic_r3i_value_handler.data(),
      r3i_value_size, r3i_dims.data(), r3i_dims.size()
  ));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, dynamic_r4i_value_handler.data(),
      r4i_value_size, r4i_dims.data(), r4i_dims.size()
  ));
  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info_handler, dynamic_dsr_value_handler.data(),
      dsr_value_size, dsr_dims.data(), dsr_dims.size()
  ));

  return input_tensors;
}

void RobustVideoMatting::detect(const cv::Mat &mat, types::MattingContent &content,
                                float downsample_ratio, bool video_mode,
                                bool remove_noise, bool minimum_post_process)
{
  if (mat.empty()) return;
  // 0. set dsr at runtime.
  dynamic_dsr_value_handler.at(0) = downsample_ratio;

  // 1. make input tensors, src, rxi, dsr
  std::vector<Ort::Value> input_tensors = this->transform(mat);
  // 2. inference, fgr, pha, rxo.
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      input_tensors.data(), num_inputs, output_node_names.data(),
      num_outputs
  );
  // 3. generate matting
  this->generate_matting(output_tensors, content, remove_noise, minimum_post_process);
  // 4. update context (needed for video detection.)
  if (video_mode)
  {
    context_is_update = false; // init state.
    this->update_context(output_tensors);
  }

}


void RobustVideoMatting::detect_video(const std::string &video_path,
                                      const std::string &output_path,
                                      std::vector<types::MattingContent> &contents,
                                      bool save_contents, float downsample_ratio,
                                      unsigned int writer_fps, bool remove_noise,
                                      bool minimum_post_process, const cv::Mat &background)
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
    this->detect(mat, content, downsample_ratio, true, remove_noise, minimum_post_process); // video_mode true
    // 3. save contents and writing out.
    if (content.flag)
    {
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
#ifdef LITEORT_DEBUG
    std::cout << i << "/" << frame_count << " done!" << "\n";
#endif
  }

  // 5. release
  video_capture.release();
  video_writer.release();

}

void RobustVideoMatting::generate_matting(std::vector<Ort::Value> &output_tensors,
                                          types::MattingContent &content,
                                          bool remove_noise,
                                          bool minimum_post_process)
{
  Ort::Value &fgr = output_tensors.at(0); // fgr (1,3,h,w) 0.~1.
  Ort::Value &pha = output_tensors.at(1); // pha (1,1,h,w) 0.~1.
  auto fgr_dims = fgr.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  auto pha_dims = pha.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int height = fgr_dims.at(2); // output height
  const unsigned int width = fgr_dims.at(3); // output width
  const unsigned int channel_step = height * width;
  // fast assign & channel transpose(CHW->HWC).
  float *fgr_ptr = fgr.GetTensorMutableData<float>();
  float *pha_ptr = pha.GetTensorMutableData<float>();
  cv::Mat rmat(height, width, CV_32FC1, fgr_ptr);
  cv::Mat gmat(height, width, CV_32FC1, fgr_ptr + channel_step);
  cv::Mat bmat(height, width, CV_32FC1, fgr_ptr + 2 * channel_step);
  cv::Mat pmat(height, width, CV_32FC1, pha_ptr); // ref only
  if (remove_noise) lite::utils::remove_small_connected_area(pmat, 0.05f);

  rmat *= 255.;
  bmat *= 255.;
  gmat *= 255.;
  std::vector<cv::Mat> fgr_channel_mats;
  fgr_channel_mats.push_back(bmat);
  fgr_channel_mats.push_back(gmat);
  fgr_channel_mats.push_back(rmat);

  // need clone to allocate a new continuous memory.
  content.pha_mat = pmat.clone(); // allocated
  cv::merge(fgr_channel_mats, content.fgr_mat); // allocated
  content.fgr_mat.convertTo(content.fgr_mat, CV_8UC3);

  if (!minimum_post_process)
  {
    cv::Mat rest = 1. - pmat;
    cv::Mat mbmat = bmat.mul(pmat) + rest * 153.;
    cv::Mat mgmat = gmat.mul(pmat) + rest * 255.;
    cv::Mat mrmat = rmat.mul(pmat) + rest * 120.;
    std::vector<cv::Mat> merge_channel_mats;
    merge_channel_mats.push_back(mbmat);
    merge_channel_mats.push_back(mgmat);
    merge_channel_mats.push_back(mrmat);
    cv::merge(merge_channel_mats, content.merge_mat); // allocated
    content.merge_mat.convertTo(content.merge_mat, CV_8UC3);
  }

  content.flag = true;
}


void RobustVideoMatting::update_context(std::vector<Ort::Value> &output_tensors)
{
  // 0. update context for video matting.
  Ort::Value &r1o = output_tensors.at(2); // fgr (1,?,?h,?w)
  Ort::Value &r2o = output_tensors.at(3); // pha (1,?,?h,?w)
  Ort::Value &r3o = output_tensors.at(4); // pha (1,?,?h,?w)
  Ort::Value &r4o = output_tensors.at(5); // pha (1,?,?h,?w)
  auto r1o_dims = r1o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  auto r2o_dims = r2o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  auto r3o_dims = r3o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  auto r4o_dims = r4o.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
  // 1. update rxi's shape according to last rxo
  dynamic_input_node_dims.at(1) = r1o_dims;
  dynamic_input_node_dims.at(2) = r2o_dims;
  dynamic_input_node_dims.at(3) = r3o_dims;
  dynamic_input_node_dims.at(4) = r4o_dims;
  // 2. update rxi's value according to last rxo
  int64_t new_r1i_value_size = this->value_size_of(r1o_dims); // (1*?*?h*?w)
  int64_t new_r2i_value_size = this->value_size_of(r2o_dims); // (1*?*?h*?w)
  int64_t new_r3i_value_size = this->value_size_of(r3o_dims); // (1*?*?h*?w)
  int64_t new_r4i_value_size = this->value_size_of(r4o_dims); // (1*?*?h*?w)
  dynamic_r1i_value_handler.resize(new_r1i_value_size);
  dynamic_r2i_value_handler.resize(new_r2i_value_size);
  dynamic_r3i_value_handler.resize(new_r3i_value_size);
  dynamic_r4i_value_handler.resize(new_r4i_value_size);
  float *new_r1i_value_ptr = r1o.GetTensorMutableData<float>();
  float *new_r2i_value_ptr = r2o.GetTensorMutableData<float>();
  float *new_r3i_value_ptr = r3o.GetTensorMutableData<float>();
  float *new_r4i_value_ptr = r4o.GetTensorMutableData<float>();
  std::memcpy(dynamic_r1i_value_handler.data(), new_r1i_value_ptr, new_r1i_value_size * sizeof(float));
  std::memcpy(dynamic_r2i_value_handler.data(), new_r2i_value_ptr, new_r2i_value_size * sizeof(float));
  std::memcpy(dynamic_r3i_value_handler.data(), new_r3i_value_ptr, new_r3i_value_size * sizeof(float));
  std::memcpy(dynamic_r4i_value_handler.data(), new_r4i_value_ptr, new_r4i_value_size * sizeof(float));
  context_is_update = true;
}

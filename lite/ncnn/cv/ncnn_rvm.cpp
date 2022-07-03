//
// Created by DefTruth on 2021/10/10.
//

#include "ncnn_rvm.h"

using ncnncv::NCNNRobustVideoMatting;


NCNNRobustVideoMatting::NCNNRobustVideoMatting(
    const std::string &_param_path, const std::string &_bin_path,
    unsigned int _num_threads, int _input_height,
    int _input_width, unsigned int _variant_type
) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads),
    input_height(_input_height), input_width(_input_width),
    variant_type(_variant_type)
{
  initialize_context();
}

void NCNNRobustVideoMatting::initialize_context()
{
  if (variant_type == VARIANT::MOBILENETV3)
  {
    if (input_width == 1920 && input_height == 1080)
    {
      r1i = ncnn::Mat(240, 135, 16); // w,h,c in NCNN
      r2i = ncnn::Mat(120, 68, 20);
      r3i = ncnn::Mat(60, 34, 40);
      r4i = ncnn::Mat(30, 17, 64);
    } // hxw 480x640 480x480 640x480
    else
    {
      r1i = ncnn::Mat(input_width / 2, input_height / 2, 16);
      r2i = ncnn::Mat(input_width / 4, input_height / 4, 20);
      r3i = ncnn::Mat(input_width / 8, input_height / 8, 40);
      r4i = ncnn::Mat(input_width / 16, input_height / 16, 64);
    }
  } // RESNET50
  else
  {
    if (input_width == 1920 && input_height == 1080)
    {
      r1i = ncnn::Mat(240, 135, 16);
      r2i = ncnn::Mat(120, 68, 32);
      r3i = ncnn::Mat(60, 34, 64);
      r4i = ncnn::Mat(30, 17, 128);
    } // hxw 480x640 480x480 640x480
    else
    {
      r1i = ncnn::Mat(input_width / 2, input_height / 2, 16);
      r2i = ncnn::Mat(input_width / 4, input_height / 4, 20);
      r3i = ncnn::Mat(input_width / 8, input_height / 8, 40);
      r4i = ncnn::Mat(input_width / 16, input_height / 16, 64);
    }
  }
  // init 0.
  r1i.fill(0.f);
  r2i.fill(0.f);
  r3i.fill(0.f);
  r4i.fill(0.f);

  context_is_initialized = true;
}

void NCNNRobustVideoMatting::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW & resize
  int h = mat.rows;
  int w = mat.cols;
  in = ncnn::Mat::from_pixels_resize(
      mat.data, ncnn::Mat::PIXEL_BGR2RGB,
      w, h, input_width, input_height
  );
  in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNRobustVideoMatting::detect(const cv::Mat &mat, types::MattingContent &content, bool video_mode)
{
  if (mat.empty()) return;
  int img_h = mat.rows;
  int img_w = mat.cols;
  if (!context_is_initialized) return;

  // 1. make input tensor
  ncnn::Mat src;
  this->transform(mat, src);

  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("src", src);
  extractor.input("r1i", r1i);
  extractor.input("r2i", r2i);
  extractor.input("r3i", r3i);
  extractor.input("r4i", r4i);

  // 3. generate matting
  this->generate_matting(extractor, content, img_h, img_w);

  // 4. update context (needed for video detection.)
  if (video_mode)
  {
    context_is_update = false; // init state.
    this->update_context(extractor);
  }
}

void NCNNRobustVideoMatting::detect_video(const std::string &video_path,
                                          const std::string &output_path,
                                          std::vector<types::MattingContent> &contents,
                                          bool save_contents, unsigned int writer_fps)
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
#ifdef LITENCNN_DEBUG
    std::cout << i << "/" << frame_count << " done!" << "\n";
#endif
  }

  // 5. release
  video_capture.release();
  video_writer.release();
}

void NCNNRobustVideoMatting::generate_matting(ncnn::Extractor &extractor,
                                              types::MattingContent &content,
                                              int img_h, int img_w)
{
  ncnn::Mat fgr, pha;
  extractor.extract("fgr", fgr);
  extractor.extract("pha", pha);
  float *fgr_ptr = (float *) fgr.data;
  float *pha_ptr = (float *) pha.data;

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

  // need clone to allocate a new continuous memory.
  content.pha_mat = pmat.clone(); // allocated
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

void NCNNRobustVideoMatting::update_context(ncnn::Extractor &extractor)
{
  ncnn::Mat r1o, r2o, r3o, r4o;
  extractor.extract("r1o", r1o);
  extractor.extract("r2o", r2o);
  extractor.extract("r3o", r3o);
  extractor.extract("r4o", r4o);

  r1i.clone_from(r1o); // deepcopy
  r2i.clone_from(r2o); // deepcopy
  r3i.clone_from(r3o); // deepcopy
  r4i.clone_from(r4o); // deepcopy

  context_is_update = true;
}

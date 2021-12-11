//
// Created by DefTruth on 2021/12/5.
//

#include "tnn_mg_matting.h"

using tnncv::TNNMGMatting;

TNNMGMatting::TNNMGMatting(
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

TNNMGMatting::~TNNMGMatting()
{
  net = nullptr;
  image_mat = nullptr;
  mask_mat = nullptr;
  instance = nullptr;
}

void TNNMGMatting::initialize_instance()
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
  image_shape = BasicTNNHandler::get_input_shape(instance, "image");
  mask_shape = BasicTNNHandler::get_input_shape(instance, "mask");

  if (image_shape.size() != 4)
  {
#ifdef LITETNN_DEBUG
    throw std::runtime_error("Found input_shape.size()!=4, but "
                             "input only support 4 dims."
                             "Such as NCHW, NHWC ...");
#else
    return;
#endif
  }
  input_mat_type = BasicTNNHandler::get_input_mat_type(instance, "image");
  input_data_format = BasicTNNHandler::get_input_data_format(instance, "image");
  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    dynamic_input_height = image_shape.at(2);
    dynamic_input_width = image_shape.at(3);
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    dynamic_input_height = image_shape.at(1);
    dynamic_input_width = image_shape.at(2);
  } // unsupport
  else
  {
#ifdef LITETNN_DEBUG
    std::cout << "input only support NCHW and NHWC "
                 "input_data_format, but found others.\n";
#endif
    return;
  }
  // 6. init output information, debug only.
  alpha_os1_shape = BasicTNNHandler::get_output_shape(instance, "alpha_os1");
  alpha_os4_shape = BasicTNNHandler::get_output_shape(instance, "alpha_os4");
  alpha_os8_shape = BasicTNNHandler::get_output_shape(instance, "alpha_os8");
#ifdef LITETNN_DEBUG
  this->print_debug_string();
#endif
}

void TNNMGMatting::print_debug_string()
{
  std::cout << "LITETNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  BasicTNNHandler::print_name_shape("image", image_shape);
  BasicTNNHandler::print_name_shape("mask", mask_shape);
  std::string data_format_string =
      (input_data_format == tnn::DATA_FORMAT_NCHW) ? "NCHW" : "NHWC";
  std::cout << "Input Data Format: " << data_format_string << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  BasicTNNHandler::print_name_shape("alpha_os1", alpha_os1_shape);
  BasicTNNHandler::print_name_shape("alpha_os4", alpha_os4_shape);
  BasicTNNHandler::print_name_shape("alpha_os8", alpha_os8_shape);
  std::cout << "========================================\n";
}

void TNNMGMatting::transform(const cv::Mat &mat, const cv::Mat &mask)
{
//  auto padded_mat = this->padding(mat); // 0-255 int8
//  auto padded_mask = this->padding(mask); // 0-1.0 float32
//  // update input mat and reshape instance
//  // reference: https://github.com/Tencent/TNN/blob/master/examples/base/ocr_text_recognizer.cc#L120
//  tnn::InputShapesMap input_shape_map;
//  BasicTNNHandler::print_name_shape("image", image_shape);
//  BasicTNNHandler::print_name_shape("mask", mask_shape);
//  std::cout << padded_mask.rows << "," << padded_mask.cols << std::endl;
//  std::cout << padded_mat.rows << "," << padded_mat.cols << std::endl;
//
//  input_shape_map.insert({"image", image_shape});
//  input_shape_map.insert({"mask", mask_shape});
//
//  auto status = instance->Reshape(input_shape_map);
//  if (status != tnn::TNN_OK)
//  {
//#ifdef LITETNN_DEBUG
//    std::cout << "instance Reshape failed in TNNMGMatting\n";
//#endif
//  }
//  std::cout << "Reshape done!" << std::endl;
//  auto new_image_shape = BasicTNNHandler::get_input_shape(instance, "image");
//  auto new_mask_shape = BasicTNNHandler::get_input_shape(instance, "mask");
//  BasicTNNHandler::print_name_shape("image", new_image_shape);
//  BasicTNNHandler::print_name_shape("mask", new_mask_shape);
//
//  cv::cvtColor(padded_mat, padded_mat, cv::COLOR_BGR2RGB);

  cv::Mat image_canvas, mask_canvas;
  cv::cvtColor(mat, image_canvas, cv::COLOR_BGR2RGB);
  cv::resize(image_canvas, image_canvas, cv::Size(dynamic_input_width, dynamic_input_height));
  cv::resize(mask, mask_canvas, cv::Size(dynamic_input_width, dynamic_input_height));

  // push into image_mat
  image_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::N8UC3,
      image_shape,
      (void *) image_canvas.data
  );
  if (!image_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "image_mat == nullptr! transform failed\n";
#endif
  }

  // push into mask_mat
  mask_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::NCHW_FLOAT,
      mask_shape,
      (void *) mask_canvas.data
  );
  if (!mask_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "mask_mat == nullptr! transform failed\n";
#endif
  }
}

cv::Mat TNNMGMatting::padding(const cv::Mat &unpad_mat)
{
  const unsigned int h = unpad_mat.rows;
  const unsigned int w = unpad_mat.cols;

  // aligned
  if (h % align_val == 0 && w % align_val == 0)
  {
    unsigned int target_h = h + 2 * align_val;
    unsigned int target_w = w + 2 * align_val;
    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, align_val,
                       align_val, align_val, cv::BORDER_REFLECT);
    return pad_mat;
  } // un-aligned
  else
  {
    // align & padding
    unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    unsigned int target_h = h + align_val + (pad_h + align_val);
    unsigned int target_w = w + align_val + (pad_w + align_val);

    cv::Mat pad_mat(target_h, target_w, unpad_mat.type());

    cv::copyMakeBorder(unpad_mat, pad_mat, align_val, pad_h + align_val,
                       align_val, pad_w + align_val, cv::BORDER_REFLECT);
    return pad_mat;
  }
}

void TNNMGMatting::update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold)
{
  if (mask.type() != CV_32FC1) mask.convertTo(mask, CV_32FC1);
  const unsigned int h = mask.rows;
  const unsigned int w = mask.cols;
  if (mask.isContinuous())
  {
    const unsigned int data_size = h * w * 1;
    float *mutable_data_ptr = (float *) mask.data;
    float guidance_threshold_ = (float) guidance_threshold;
    for (unsigned int i = 0; i < data_size; ++i)
    {
      if (mutable_data_ptr[i] >= guidance_threshold_)
        mutable_data_ptr[i] = 1.0f;
      else
        mutable_data_ptr[i] = 0.0f;
    }
  } //
  else
  {
    float guidance_threshold_ = (float) guidance_threshold;
    for (unsigned int i = 0; i < h; ++i)
    {
      float *p = mask.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
      {
        if (p[j] >= guidance_threshold_)
          p[j] = 1.0;
        else
          p[j] = 0.;
      }
    }
  }
}

void TNNMGMatting::detect(const cv::Mat &mat, cv::Mat &mask, types::MattingContent &content,
                          bool remove_noise, unsigned int guidance_threshold)
{
  if (mat.empty() || mask.empty()) return;
  // const unsigned int img_height = mat.rows;
  // const unsigned int img_width = mat.cols;
  // this->update_dynamic_shape(img_height, img_width);
  this->update_guidance_mask(mask, guidance_threshold); // -> float32 hw1 0~1.0

  // 1. make input tensors, image, mask
  this->transform(mat, mask);

  // 2. set input_mat
  tnn::MatConvertParam image_cvt_param, mask_cvt_param;
  image_cvt_param.scale = scale_vals;
  image_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(image_mat, image_cvt_param, "image");
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }
  status = instance->SetInputMat(mask_mat, mask_cvt_param, "mask");
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
  this->generate_matting(instance, mat, content, remove_noise);
}

void TNNMGMatting::generate_matting(
    std::shared_ptr<tnn::Instance> &_instance,
    const cv::Mat &mat, types::MattingContent &content,
    bool remove_noise)
{
  std::shared_ptr<tnn::Mat> alpha_os1_mat;
  std::shared_ptr<tnn::Mat> alpha_os4_mat;
  std::shared_ptr<tnn::Mat> alpha_os8_mat;
  tnn::MatConvertParam cvt_param;
  tnn::Status status_os1, status_os4, status_os8;

  // https://github.com/yucornetto/MGMatting/blob/main/code-base/infer.py
  // e.g (1,1,h+2*pad_val,w+2*pad_val)
  status_os1 = _instance->GetOutputMat(alpha_os1_mat, cvt_param, "alpha_os1", output_device_type);
  status_os4 = _instance->GetOutputMat(alpha_os4_mat, cvt_param, "alpha_os4", output_device_type);
  status_os8 = _instance->GetOutputMat(alpha_os8_mat, cvt_param, "alpha_os8", output_device_type);
  if (status_os1 != tnn::TNN_OK || status_os4 != tnn::TNN_OK || status_os8 != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance->GetOutputMat failed!:"
              << status_os1.description().c_str() << ": "
              << status_os4.description().c_str() << ": "
              << status_os8.description().c_str() << "\n";
#endif
    return;
  }

  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = alpha_os1_mat->GetDims();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  float *alpha_os1_ptr = (float *) alpha_os1_mat->GetData();
  float *alpha_os4_ptr = (float *) alpha_os4_mat->GetData();
  float *alpha_os8_ptr = (float *) alpha_os8_mat->GetData();

  cv::Mat alpha_os1_pred(out_h, out_w, CV_32FC1, alpha_os1_ptr);
  cv::Mat alpha_os4_pred(out_h, out_w, CV_32FC1, alpha_os4_ptr);
  cv::Mat alpha_os8_pred(out_h, out_w, CV_32FC1, alpha_os8_ptr);

  cv::Mat alpha_pred(out_h, out_w, CV_32FC1, alpha_os8_ptr);
  cv::Mat weight_os4 = this->get_unknown_tensor_from_pred(alpha_pred, 30);
  this->update_alpha_pred(alpha_pred, weight_os4, alpha_os4_pred);
  cv::Mat weight_os1 = this->get_unknown_tensor_from_pred(alpha_pred, 15);
  this->update_alpha_pred(alpha_pred, weight_os1, alpha_os1_pred);
  // post process
  if (remove_noise) this->remove_small_connected_area(alpha_pred);

  cv::Mat mat_copy;
  mat.convertTo(mat_copy, CV_32FC3);
  cv::Mat pmat = alpha_pred;

  if (out_h != h || out_w != w) cv::resize(pmat, pmat, cv::Size(w, h));

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

// https://github.com/yucornetto/MGMatting/issues/11
// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L225
cv::Mat TNNMGMatting::get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width)
{
  const unsigned int h = alpha_pred.rows;
  const unsigned int w = alpha_pred.cols;
  const unsigned int data_size = h * w;
  cv::Mat uncertain_area(h, w, CV_32FC1, cv::Scalar(1.0f)); // continuous
  const float *pred_ptr = (float *) alpha_pred.data;
  float *uncertain_ptr = (float *) uncertain_area.data;
  // threshold
  if (alpha_pred.isContinuous() && uncertain_area.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if ((pred_ptr[i] < 1.0f / 255.0f) || (pred_ptr[i] > 1.0f - 1.0f / 255.0f))
        uncertain_ptr[i] = 0.f;
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      const float *pred_row_ptr = alpha_pred.ptr<float>(i);
      float *uncertain_row_ptr = uncertain_area.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
      {
        if ((pred_row_ptr[j] < 1.0f / 255.0f) || (pred_row_ptr[j] > 1.0f - 1.0f / 255.0f))
          uncertain_row_ptr[j] = 0.f;
      }
    }
  }
  // dilate
  unsigned int size = rand_width / 2;
  auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size, size));
  cv::dilate(uncertain_area, uncertain_area, kernel);

  // weight
  cv::Mat weight(h, w, CV_32FC1, uncertain_area.data); // ref only, zero copy.
  float *weight_ptr = (float *) weight.data;
  if (weight.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if (weight_ptr[i] != 1.0f) weight_ptr[i] = 0;
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      float *weight_row_ptr = weight.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
        if (weight_row_ptr[j] != 1.0f) weight_row_ptr[j] = 0.f;

    }
  }

  return weight;
}

void TNNMGMatting::update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred)
{
  const unsigned int h = alpha_pred.rows;
  const unsigned int w = alpha_pred.cols;
  const unsigned int data_size = h * w;
  const float *weight_ptr = (float *) weight.data;
  float *mutable_alpha_ptr = (float *) alpha_pred.data;
  const float *other_alpha_ptr = (float *) other_alpha_pred.data;

  if (alpha_pred.isContinuous() && weight.isContinuous() && other_alpha_pred.isContinuous())
  {
    for (unsigned int i = 0; i < data_size; ++i)
      if (weight_ptr[i] > 0.f) mutable_alpha_ptr[i] = other_alpha_ptr[i];
  } //
  else
  {
    for (unsigned int i = 0; i < h; ++i)
    {
      const float *weight_row_ptr = weight.ptr<float>(i);
      float *mutable_alpha_row_ptr = alpha_pred.ptr<float>(i);
      const float *other_alpha_row_ptr = other_alpha_pred.ptr<float>(i);
      for (unsigned int j = 0; j < w; ++j)
        if (weight_row_ptr[j] > 0.f) mutable_alpha_row_ptr[j] = other_alpha_row_ptr[j];
    }
  }
}

// https://github.com/yucornetto/MGMatting/blob/main/code-base/utils/util.py#L208
void TNNMGMatting::remove_small_connected_area(cv::Mat &alpha_pred)
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

void TNNMGMatting::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
{
  // update dynamic input dims
  unsigned int h = img_height;
  unsigned int w = img_width;
  // update dynamic input dims
  if (h % align_val == 0 && w % align_val == 0)
  {
    // aligned
    dynamic_input_height = h + 2 * align_val;
    dynamic_input_width = w + 2 * align_val;
  } // un-aligned
  else
  {
    // align first
    unsigned int align_h = align_val * ((h - 1) / align_val + 1);
    unsigned int align_w = align_val * ((w - 1) / align_val + 1);
    unsigned int pad_h = align_h - h; // >= 0
    unsigned int pad_w = align_w - w; // >= 0
    dynamic_input_height = h + align_val + (pad_h + align_val);
    dynamic_input_width = w + align_val + (pad_w + align_val);
  }

  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    image_shape.at(2) = dynamic_input_height;
    image_shape.at(3) = dynamic_input_width;
    mask_shape.at(2) = dynamic_input_height;
    mask_shape.at(3) = dynamic_input_width;
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    image_shape.at(1) = dynamic_input_height;
    image_shape.at(2) = dynamic_input_width;
    mask_shape.at(1) = dynamic_input_height;
    mask_shape.at(2) = dynamic_input_width;
  }
}
//
// Created by DefTruth on 2021/12/5.
//

#include "mnn_mg_matting.h"

using mnncv::MNNMGMatting;

MNNMGMatting::MNNMGMatting(
    const std::string &_mnn_path, unsigned int _num_threads
) : log_id(_mnn_path.data()),
    mnn_path(_mnn_path.data()),
    num_threads(_num_threads)
{
  initialize_interpreter();
  initialize_pretreat();
}

MNNMGMatting::~MNNMGMatting()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNMGMatting::initialize_pretreat()
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

void MNNMGMatting::initialize_interpreter()
{
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  // 2. init schedule_config
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  // 3. create session
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // 4. init input tensor
  image_tensor = mnn_interpreter->getSessionInput(mnn_session, "image");
  mask_tensor = mnn_interpreter->getSessionInput(mnn_session, "mask");
  dimension_type = image_tensor->getDimensionType(); // CAFFE(NCHW)
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

void MNNMGMatting::transform(const cv::Mat &mat, const cv::Mat &mask)
{
  auto padded_mat = this->padding(mat); // 0-255 int8
  auto padded_mask = this->padding(mask); // 0-1.0 float32

  // update input tensor and resize Session
  mnn_interpreter->resizeTensor(image_tensor, {1, 3, dynamic_input_height, dynamic_input_width});
  mnn_interpreter->resizeTensor(mask_tensor, {1, 1, dynamic_input_height, dynamic_input_width});
  mnn_interpreter->resizeSession(mnn_session);

  // push data into image tensor
  pretreat->convert(padded_mat.data, dynamic_input_width, dynamic_input_height,
                    padded_mat.step[0], image_tensor);

  // push data into mask tensor
  auto tmp_host_nchw_tensor = new MNN::Tensor(mask_tensor, MNN::Tensor::CAFFE); // tmp
  std::memcpy(tmp_host_nchw_tensor->host<float>(), padded_mask.data,
              dynamic_input_mask_size * sizeof(float));
  mask_tensor->copyFromHostTensor(tmp_host_nchw_tensor);

  delete tmp_host_nchw_tensor;
}

cv::Mat MNNMGMatting::padding(const cv::Mat &unpad_mat)
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

void MNNMGMatting::update_guidance_mask(cv::Mat &mask, unsigned int guidance_threshold)
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

void MNNMGMatting::detect(const cv::Mat &mat, cv::Mat &mask, types::MattingContent &content,
                          bool remove_noise, unsigned int guidance_threshold)
{
  if (mat.empty() || mask.empty()) return;
  const unsigned int img_height = mat.rows;
  const unsigned int img_width = mat.cols;
  this->update_dynamic_shape(img_height, img_width);
  this->update_guidance_mask(mask, guidance_threshold); // -> float32 hw1 0~1.0

  // 1. make input tensors, image, mask
  this->transform(mat, mask);
  // 2. inference & run session
  mnn_interpreter->runSession(mnn_session);

  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate matting
  this->generate_matting(output_tensors, mat, content, remove_noise);
}

void MNNMGMatting::generate_matting(
    const std::map<std::string, MNN::Tensor *> &output_tensors,
    const cv::Mat &mat, types::MattingContent &content,
    bool remove_noise)
{
  // https://github.com/yucornetto/MGMatting/blob/main/code-base/infer.py
  auto device_alpha_os1_ptr = output_tensors.at("alpha_os1"); // e.g (1,1,h+2*pad_val,w+2*pad_val)
  auto device_alpha_os4_ptr = output_tensors.at("alpha_os4"); // e.g (1,1,h+2*pad_val,w+2*pad_val)
  auto device_alpha_os8_ptr = output_tensors.at("alpha_os8"); // e.g (1,1,h+2*pad_val,w+2*pad_val)
  MNN::Tensor host_alpha_os1_tensor(device_alpha_os1_ptr, device_alpha_os1_ptr->getDimensionType());
  MNN::Tensor host_alpha_os4_tensor(device_alpha_os4_ptr, device_alpha_os4_ptr->getDimensionType());
  MNN::Tensor host_alpha_os8_tensor(device_alpha_os8_ptr, device_alpha_os8_ptr->getDimensionType());
  device_alpha_os1_ptr->copyToHostTensor(&host_alpha_os1_tensor);
  device_alpha_os4_ptr->copyToHostTensor(&host_alpha_os4_tensor);
  device_alpha_os8_ptr->copyToHostTensor(&host_alpha_os8_tensor);

  const unsigned int h = mat.rows;
  const unsigned int w = mat.cols;

  auto output_dims = host_alpha_os1_tensor.shape();
  const unsigned int out_h = output_dims.at(2);
  const unsigned int out_w = output_dims.at(3);
  float *alpha_os1_ptr = host_alpha_os1_tensor.host<float>();
  float *alpha_os4_ptr = host_alpha_os4_tensor.host<float>();
  float *alpha_os8_ptr = host_alpha_os8_tensor.host<float>();

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
  // cv::Mat pred_alpha_mat(out_h, out_w, CV_32FC1, alpha_os1_ptr);
  cv::Mat pmat = alpha_pred(cv::Rect(align_val, align_val, w, h));

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
cv::Mat MNNMGMatting::get_unknown_tensor_from_pred(const cv::Mat &alpha_pred, unsigned int rand_width)
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

void MNNMGMatting::update_alpha_pred(cv::Mat &alpha_pred, const cv::Mat &weight, const cv::Mat &other_alpha_pred)
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
void MNNMGMatting::remove_small_connected_area(cv::Mat &alpha_pred)
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

void MNNMGMatting::update_dynamic_shape(unsigned int img_height, unsigned int img_width)
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

  dynamic_input_image_size = 1 * 3 * dynamic_input_height * dynamic_input_width;
  dynamic_input_mask_size = 1 * 1 * dynamic_input_height * dynamic_input_width;
}

void MNNMGMatting::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (image_tensor) image_tensor->printShape();
  if (mask_tensor) mask_tensor->printShape();
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
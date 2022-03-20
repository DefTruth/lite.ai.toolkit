//
// Created by DefTruth on 2022/3/20.
//

#include "tnn_pipnet98.h"

using tnncv::TNNPIPNet98;

TNNPIPNet98::TNNPIPNet98(const std::string &_proto_path,
                         const std::string &_model_path,
                         unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

void TNNPIPNet98::transform(const cv::Mat &mat_rs)
{
  // be carefully, no deepcopy inside this tnn::Mat constructor,
  // so, we can not pass a local cv::Mat to this constructor.
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNPIPNet98::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input mat
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  this->transform(mat_rs); // resize outside transform to prevent overflow
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
  // 3. generate landmarks
  this->generate_landmarks(landmarks, instance, img_height, img_width);
}

void TNNPIPNet98::generate_landmarks(types::Landmarks &landmarks,
                                     std::shared_ptr<tnn::Instance> &_instance,
                                     float img_height, float img_width)
{
  std::shared_ptr<tnn::Mat> outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y;
  tnn::MatConvertParam cvt_param;
  tnn::Status status_cls = _instance->GetOutputMat(outputs_cls, cvt_param, "outputs_cls", output_device_type);
  tnn::Status status_x = _instance->GetOutputMat(outputs_x, cvt_param, "outputs_x", output_device_type);
  tnn::Status status_y = _instance->GetOutputMat(outputs_y, cvt_param, "outputs_y", output_device_type);
  tnn::Status status_nb_x = _instance->GetOutputMat(outputs_nb_x, cvt_param, "outputs_nb_x", output_device_type);
  tnn::Status status_nb_y = _instance->GetOutputMat(outputs_nb_y, cvt_param, "outputs_nb_y", output_device_type);

  if (status_cls != tnn::TNN_OK || status_x != tnn::TNN_OK || status_y != tnn::TNN_OK
      || status_nb_x != tnn::TNN_OK || status_nb_y != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status_cls.description().c_str() << ": "
              << status_x.description().c_str() << ": "
              << status_y.description().c_str() << ": "
              << status_nb_x.description().c_str() << ": "
              << status_nb_y.description().c_str() << "\n";
#endif
    return;
  }
  auto cls_shape = outputs_cls->GetDims();
  const unsigned int grid_h = cls_shape.at(2); // 8
  const unsigned int grid_w = cls_shape.at(3); // 8
  const unsigned int grid_length = grid_h * grid_w; // 8 * 8 = 64
  const unsigned int input_h = input_height;
  const unsigned int input_w = input_width;

  // fetch data from pointers
  const float *outputs_cls_ptr = (float *) outputs_cls->GetData();
  const float *outputs_x_ptr = (float *) outputs_x->GetData();
  const float *outputs_y_ptr = (float *) outputs_y->GetData();
  const float *outputs_nb_x_ptr = (float *) outputs_nb_x->GetData();
  const float *outputs_nb_y_ptr = (float *) outputs_nb_y->GetData();

  // find max_ids
  std::vector<unsigned int> max_ids(num_lms);
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    const float *score_ptr = outputs_cls_ptr + i * grid_length;
    unsigned int max_id = 0;
    float max_score = score_ptr[0];
    for (unsigned int j = 0; j < grid_length; ++j)
    {
      if (score_ptr[j] > max_score)
      {
        max_score = score_ptr[j];
        max_id = j;
      }
    }
    max_ids[i] = max_id; // range 0~64
  }

  // find x & y offsets
  std::vector<float> output_x_select(num_lms);
  std::vector<float> output_y_select(num_lms);
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    const float *offset_x_ptr = outputs_x_ptr + i * grid_length;
    const float *offset_y_ptr = outputs_y_ptr + i * grid_length;
    const unsigned int max_id = max_ids.at(i);
    output_x_select[i] = offset_x_ptr[max_id];
    output_y_select[i] = offset_y_ptr[max_id];
  }

  // find nb_x & nb_y offsets
  std::unordered_map<unsigned int, std::vector<float>> output_nb_x_select;
  std::unordered_map<unsigned int, std::vector<float>> output_nb_y_select;
  // initialize offsets map
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    std::vector<float> nb_x_offset(num_nb);
    std::vector<float> nb_y_offset(num_nb);
    output_nb_x_select[i] = nb_x_offset;
    output_nb_x_select[i] = nb_y_offset;
  }
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    for (unsigned int j = 0; j < num_nb; ++j)
    {
      const float *offset_nb_x_ptr = outputs_nb_x_ptr + (i * num_nb + j) * grid_length;
      const float *offset_nb_y_ptr = outputs_nb_y_ptr + (i * num_nb + j) * grid_length;
      const unsigned int max_id = max_ids.at(i);
      output_nb_x_select[i][j] = offset_nb_x_ptr[max_id];
      output_nb_y_select[i][j] = offset_nb_y_ptr[max_id];
    }
  }

  // calculate coords
  std::vector<float> lms_pred_x(num_lms); // 98
  std::vector<float> lms_pred_y(num_lms); // 98
  std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_x; // 98,10
  std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_y; // 98,10
  // initialize pred maps
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    std::vector<float> nb_x_offset(num_nb);
    std::vector<float> nb_y_offset(num_nb);
    lms_pred_nb_x[i] = nb_x_offset;
    lms_pred_nb_y[i] = nb_y_offset;
  }
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    float cx = static_cast<float>(max_ids.at(i) % grid_w);
    float cy = static_cast<float>(max_ids.at(i) / grid_w);
    // calculate coords & normalize
    lms_pred_x[i] = ((cx + output_x_select[i]) * (float) net_stride) / (float) input_w;
    lms_pred_y[i] = ((cy + output_y_select[i]) * (float) net_stride) / (float) input_h;
    for (unsigned int j = 0; j < num_nb; ++j)
    {
      lms_pred_nb_x[i][j] = ((cx + output_nb_x_select[i][j]) * (float) net_stride) / (float) input_w;
      lms_pred_nb_y[i][j] = ((cy + output_nb_y_select[i][j]) * (float) net_stride) / (float) input_h;
    }
  }

  // reverse indexes
  std::unordered_map<unsigned int, std::vector<float>> tmp_nb_x; // 98,max_len
  std::unordered_map<unsigned int, std::vector<float>> tmp_nb_y; // 98,max_len
  // initialize reverse maps
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    std::vector<float> tmp_x(max_len);
    std::vector<float> tmp_y(max_len);
    tmp_nb_x[i] = tmp_x;
    tmp_nb_y[i] = tmp_y;
  }
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    for (unsigned int j = 0; j < max_len; ++j)
    {
      unsigned int ri = reverse_index1[i * max_len + j];
      unsigned int rj = reverse_index2[i * max_len + j];
      tmp_nb_x[i][j] = lms_pred_nb_x[ri][rj];
      tmp_nb_y[i][j] = lms_pred_nb_y[ri][rj];
    }
  }

  // merge predictions
  landmarks.points.clear();
  for (unsigned int i = 0; i < num_lms; ++i)
  {
    float total_x = lms_pred_x[i];
    float total_y = lms_pred_y[i];
    for (unsigned int j = 0; j < max_len; ++j)
    {
      total_x += tmp_nb_x[i][j];
      total_y += tmp_nb_y[i][j];
    }
    float x = total_x / ((float) max_len + 1.f);
    float y = total_y / ((float) max_len + 1.f);
    x = std::min(std::max(0.f, x), 1.0f);
    y = std::min(std::max(0.f, y), 1.0f);

    landmarks.points.push_back(cv::Point2f(x * img_width, y * img_height));
  }

  landmarks.flag = true;
}



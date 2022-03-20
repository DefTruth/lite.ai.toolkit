//
// Created by DefTruth on 2022/3/20.
//

#include "pipnet98.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::PIPNet98;


Ort::Value PIPNet98::transform(const cv::Mat &mat)
{
  cv::Mat canvas;
  cv::resize(mat, canvas, cv::Size(input_node_dims.at(3),
                                   input_node_dims.at(2)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f);
  ortcv::utils::transform::normalize_inplace(canvas, mean_vals, scale_vals); // float32
  // (1,3,256,256)
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW); // deepcopy inside
}

void PIPNet98::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate landmarks
  this->generate_landmarks(landmarks, output_tensors, img_height, img_width);
}

void PIPNet98::generate_landmarks(types::Landmarks &landmarks,
                                  std::vector<Ort::Value> &output_tensors,
                                  float img_height, float img_width)
{
  Ort::Value &outputs_cls = output_tensors.at(0);  // (1,98,8,8)
  Ort::Value &outputs_x = output_tensors.at(1);  // (1,98,8,8)
  Ort::Value &outputs_y = output_tensors.at(2);  // (1,98,8,8)
  Ort::Value &outputs_nb_x = output_tensors.at(3);  // (1,98*10,8,8)
  Ort::Value &outputs_nb_y = output_tensors.at(4);  // (1,98*10,8,8)
  const unsigned int grid_h = output_node_dims.at(0).at(2); // 8
  const unsigned int grid_w = output_node_dims.at(0).at(3); // 8
  const unsigned int grid_length = grid_h * grid_w; // 8 * 8 = 64
  const unsigned int input_h = input_node_dims.at(2);
  const unsigned int input_w = input_node_dims.at(3);

  // fetch data from pointers
  const float *outputs_cls_ptr = outputs_cls.GetTensorMutableData<float>();
  const float *outputs_x_ptr = outputs_x.GetTensorMutableData<float>();
  const float *outputs_y_ptr = outputs_y.GetTensorMutableData<float>();
  const float *outputs_nb_x_ptr = outputs_nb_x.GetTensorMutableData<float>();
  const float *outputs_nb_y_ptr = outputs_nb_y.GetTensorMutableData<float>();

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




//
// Created by DefTruth on 2022/3/20.
//

#include "ncnn_pipnet68.h"

using ncnncv::NCNNPIPNet68;

NCNNPIPNet68::NCNNPIPNet68(const std::string &_param_path,
                           const std::string &_bin_path,
                           unsigned int _num_threads) :
    BasicNCNNHandler(_param_path, _bin_path, _num_threads)
{
}

void NCNNPIPNet68::transform(const cv::Mat &mat, ncnn::Mat &in)
{
  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_width, input_height));
  // will do deepcopy inside ncnn
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
}


void NCNNPIPNet68::detect(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (mat.empty()) return;
  float img_height = static_cast<float>(mat.rows);
  float img_width = static_cast<float>(mat.cols);

  // 1. make input tensor
  ncnn::Mat input;
  this->transform(mat, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("img", input);
  // 3. generate landmarks
  this->generate_landmarks(landmarks, extractor, img_height, img_width);
}

void NCNNPIPNet68::generate_landmarks(types::Landmarks &landmarks,
                                      ncnn::Extractor &extractor,
                                      float img_height, float img_width)
{
  ncnn::Mat outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y;
  extractor.extract("outputs_cls", outputs_cls); // (68,8,8)
  extractor.extract("outputs_x", outputs_x); // (68,8,8)
  extractor.extract("outputs_y", outputs_y); // (68,8,8)
  extractor.extract("outputs_nb_x", outputs_nb_x); // (68*10,8,8)
  extractor.extract("outputs_nb_y", outputs_nb_y); // (68*10,8,8)
#ifdef LITENCNN_DEBUG
  BasicNCNNHandler::print_shape(outputs_cls, "outputs_cls");
  BasicNCNNHandler::print_shape(outputs_x, "outputs_x");
  BasicNCNNHandler::print_shape(outputs_y, "outputs_y");
  BasicNCNNHandler::print_shape(outputs_nb_x, "outputs_nb_x");
  BasicNCNNHandler::print_shape(outputs_nb_y, "outputs_nb_y");
#endif
  const unsigned int input_h = input_height;
  const unsigned int input_w = input_width;

  // fetch data from pointers
  const float *outputs_cls_ptr = (float *) outputs_cls.data;
  const float *outputs_x_ptr = (float *) outputs_x.data;
  const float *outputs_y_ptr = (float *) outputs_y.data;
  const float *outputs_nb_x_ptr = (float *) outputs_nb_x.data;
  const float *outputs_nb_y_ptr = (float *) outputs_nb_y.data;

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
    output_nb_y_select[i] = nb_y_offset;
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
  std::vector<float> lms_pred_x(num_lms); // 68
  std::vector<float> lms_pred_y(num_lms); // 68
  std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_x; // 68,10
  std::unordered_map<unsigned int, std::vector<float>> lms_pred_nb_y; // 68,10
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
  std::unordered_map<unsigned int, std::vector<float>> tmp_nb_x; // 68,max_len
  std::unordered_map<unsigned int, std::vector<float>> tmp_nb_y; // 68,max_len
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
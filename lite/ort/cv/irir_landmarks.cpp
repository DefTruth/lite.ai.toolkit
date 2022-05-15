//
// Created by DefTruth on 2022/5/2.
//

#include "iris_landmarks.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::IrisLandmarks;

Ort::Value IrisLandmarks::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  // (1,3,64,64) 1xCXHXW
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::CHW);
}

void IrisLandmarks::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                   int target_height, int target_width,
                                   IrisScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(0, 0, 0));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat = mat.clone();
  cv::resize(new_unpad_mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void IrisLandmarks::detect(const cv::Mat &mat, types::Landmarks3D &eyes_contours_and_brows,
                           types::Landmarks3D &iris)
{
  if (mat.empty()) return;
  const int input_height = input_node_dims.at(2);
  const int input_width = input_node_dims.at(3);
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  IrisScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference landmarks3d
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate landmarks3d
  this->generate_eye_contours_brows_and_iris_landmarks3d(
      scale_params, output_tensors, eyes_contours_and_brows, iris,
      img_height, img_width);
}

void IrisLandmarks::generate_eye_contours_brows_and_iris_landmarks3d(
    const IrisScaleParams &scale_params,
    std::vector<Ort::Value> &output_tensors,
    types::Landmarks3D &eyes_contours_and_brows,
    types::Landmarks3D &iris,
    int img_height, int img_width)
{
  Ort::Value &eyes_contours_and_brows_pred = output_tensors.at(0); // (1,213=71*3)
  Ort::Value &iris_pred = output_tensors.at(1); // (1,15=5*3)

  auto iris_dims = iris_pred.GetTensorTypeAndShapeInfo().GetShape();
  auto eyes_contours_and_brows_dims =
      eyes_contours_and_brows_pred.GetTensorTypeAndShapeInfo().GetShape();
  const unsigned int num_iris_element = iris_dims.at(1); // 15=5*3
  const unsigned int num_eyes_contours_brows_element = eyes_contours_and_brows_dims.at(1); // 213=71*3
  const float *iris_ptr = iris_pred.GetTensorMutableData<float>();
  const float *eyes_contours_brows_ptr = eyes_contours_and_brows_pred.GetTensorMutableData<float>();

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  eyes_contours_and_brows.points.clear();
  // fetch non-normalized eyes_contours_and_brows 71 points with target size (64)
  for (unsigned int i = 0; i < num_eyes_contours_brows_element; i += 3)
  {
    cv::Point3f point3d;
    point3d.x = (eyes_contours_brows_ptr[i] - (float) dw_) / r_;
    point3d.y = (eyes_contours_brows_ptr[i + 1] - (float) dh_) / r_;
    point3d.z = (eyes_contours_brows_ptr[i + 2] / r_);
    // border clamp
    point3d.x = std::min(std::max(0.f, point3d.x), (float) img_width - 1.f);
    point3d.y = std::min(std::max(0.f, point3d.y), (float) img_height - 1.f);
    eyes_contours_and_brows.points.push_back(point3d);
  }
  eyes_contours_and_brows.flag = true;

#if LITEORT_DEBUG
  std::cout << "detected num_eyes_contours_brows_element: "
            << num_eyes_contours_brows_element << "\n";
  std::cout << "generate eyes_contours_and_brows landmarks3d num: "
            << eyes_contours_and_brows.points.size() << "\n";
#endif

  iris.points.clear();
  // fetch non-normalized iris 5 points with target size (64)
  for (unsigned int i = 0; i < num_iris_element; i += 3)
  {
    cv::Point3f point3d;
    point3d.x = (iris_ptr[i] - (float) dw_) / r_;
    point3d.y = (iris_ptr[i + 1] - (float) dh_) / r_;
    point3d.z = (iris_ptr[i + 2] / r_);
    // border clamp
    point3d.x = std::min(std::max(0.f, point3d.x), (float) img_width - 1.f);
    point3d.y = std::min(std::max(0.f, point3d.y), (float) img_height - 1.f);
    iris.points.push_back(point3d);
  }
  iris.flag = true;

#if LITEORT_DEBUG
  std::cout << "detected iris: " << num_iris_element << "\n";
  std::cout << "generate iris landmarks3d num: " << iris.points.size() << "\n";
#endif
}
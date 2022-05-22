//
// Created by DefTruth on 2022/5/2.
//

#include "facemesh.h"
#include "lite/ort/core/ort_utils.h"

using ortcv::FaceMesh;

Ort::Value FaceMesh::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  // (1,192,192,3) 1xHXWXC
  ortcv::utils::transform::normalize_inplace(canvas, mean_val, scale_val); // float32
  return ortcv::utils::transform::create_tensor(
      canvas, input_node_dims, memory_info_handler,
      input_values_handler, ortcv::utils::transform::HWC);
}

void FaceMesh::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                              int target_height, int target_width,
                              FaceMeshScaleParams &scale_params)
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

void FaceMesh::detect(const cv::Mat &mat, types::Landmarks3D &landmarks3d, float &confidence)
{
  if (mat.empty()) return;
  // (1,192,192,3)
  const int input_height = input_node_dims.at(1);
  const int input_width = input_node_dims.at(2);
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  FaceMeshScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  Ort::Value input_tensor = this->transform(mat_rs);
  // 2. inference landmarks3d & confidence
  auto output_tensors = ort_session->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(),
      &input_tensor, 1, output_node_names.data(), num_outputs
  );
  // 3. generate landmarks3d and face presence confidence
  this->generate_landmarks3d_and_confidence(
      scale_params, output_tensors, landmarks3d,
      confidence, img_height, img_width);
}

static inline float sigmoid(float x)
{ return (float) (1.f / (1.f + std::exp(-x))); }

void FaceMesh::generate_landmarks3d_and_confidence(const FaceMeshScaleParams &scale_params,
                                                   std::vector<Ort::Value> &output_tensors,
                                                   types::Landmarks3D &landmarks3d,
                                                   float &confidence, int img_height,
                                                   int img_width)
{
  Ort::Value &landmarks_pred = output_tensors.at(0); // (1,1,1,1404=468*3)
  Ort::Value &confidence_pred = output_tensors.at(1); // (1,)

  auto output_dims = landmarks_pred.GetTensorTypeAndShapeInfo().GetShape(); // (1,1,1,1404=468*3)
  const unsigned int num_element = output_dims.at(3); // 1404
  const float *landmarks_ptr = landmarks_pred.GetTensorMutableData<float>();
  const float *confidence_ptr = confidence_pred.GetTensorMutableData<float>();

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  confidence = sigmoid(confidence_ptr[0]);

  landmarks3d.points.clear();
  // fetch non-normalized 468 points with target size (192)
  for (unsigned int i = 0; i < num_element; i += 3)
  {
    cv::Point3f point3d;
    point3d.x = (landmarks_ptr[i] - (float) dw_) / r_;
    point3d.y = (landmarks_ptr[i + 1] - (float) dh_) / r_;
    point3d.z = (landmarks_ptr[i + 2] / r_);
    // border clamp
    point3d.x = std::min(std::max(0.f, point3d.x), (float) img_width - 1.f);
    point3d.y = std::min(std::max(0.f, point3d.y), (float) img_height - 1.f);
    landmarks3d.points.push_back(point3d);
  }
  landmarks3d.flag = true;

#if LITEORT_DEBUG
  std::cout << "detected num_element: " << num_element << "\n";
  std::cout << "generate landmarks3d num: " << landmarks3d.points.size() << "\n";
#endif
}

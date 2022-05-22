//
// Created by DefTruth on 2022/5/15.
//

#include "mnn_iris_landmarks.h"

using mnncv::MNNIrisLandmarks;

MNNIrisLandmarks::MNNIrisLandmarks(const std::string &_mnn_path, unsigned int _num_threads) :
    mnn_path(_mnn_path.data()), log_id(_mnn_path.data()), num_threads(_num_threads)
{
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // resize tensor & session (NHWC) (1,64,64,3)
  input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
  mnn_interpreter->resizeTensor(
      input_tensor, {input_batch, input_height, input_width, input_channel});
  mnn_interpreter->resizeSession(mnn_session); // may not need
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

MNNIrisLandmarks::~MNNIrisLandmarks() noexcept
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNIrisLandmarks::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (input_tensor) input_tensor->printShape();
  auto dimension_type = input_tensor->getDimensionType();
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

void MNNIrisLandmarks::transform(cv::Mat &mat_rs)
{
  cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
  mat_rs.convertTo(mat_rs, CV_32FC3, 1.f / 255.f, 0.f);

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NullDereference"
  MNN::Tensor tmp_host_tensor(input_tensor, input_tensor->getDimensionType());
#pragma clang diagnostic pop

  std::memcpy(tmp_host_tensor.host<float>(), (void *) mat_rs.data,
              3 * input_height * input_width * sizeof(float));
  input_tensor->copyFromHostTensor(&tmp_host_tensor);
}

void MNNIrisLandmarks::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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

void MNNIrisLandmarks::detect(const cv::Mat &mat, types::Landmarks3D &eyes_contours_and_brows,
                              types::Landmarks3D &iris, bool is_screen_right_eye)
{
  if (mat.empty()) return;

  cv::Mat mat_ref;
  if (is_screen_right_eye)
    cv::flip(mat, mat_ref, 1);
  else mat_ref = mat; // ref only.

  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  IrisScaleParams scale_params;
  this->resize_unscale(mat_ref, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  this->transform(mat_rs);
  //  2. inference landmarks3d
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);

  // 3. generate landmarks3d
  this->generate_eye_contours_brows_and_iris_landmarks3d(
      scale_params, output_tensors, eyes_contours_and_brows, iris,
      img_height, img_width);

  // horizontal flip the detected points if source image is
  // screen right eye according to target size (64, 64)
  if (is_screen_right_eye && eyes_contours_and_brows.flag && iris.flag)
  {
    for (auto &point3d: eyes_contours_and_brows.points)
      point3d.x = (float) (img_width - 1.f) - point3d.x;
    for (auto &point3d: iris.points)
      point3d.x = (float) (img_width - 1.f) - point3d.x;
  }

#if LITEMNN_DEBUG
  std::cout << "IrisLandmarks is_screen_right_eye: " << is_screen_right_eye << "\n";
#endif
}

void MNNIrisLandmarks::generate_eye_contours_brows_and_iris_landmarks3d(
    const IrisScaleParams &scale_params,
    const std::map<std::string, MNN::Tensor *> &output_tensors,
    types::Landmarks3D &eyes_contours_and_brows,
    types::Landmarks3D &iris, int img_height, int img_width)
{
  auto device_eyes_contours_and_brows = output_tensors.at("output_eyes_contours_and_brows"); // (1,213)
  auto device_iris = output_tensors.at("output_iris"); // (1,15)
  auto dimension_type = device_eyes_contours_and_brows->getDimensionType();
  MNN::Tensor host_eyes_contours_and_brows(device_eyes_contours_and_brows, dimension_type);
  MNN::Tensor host_iris(device_iris, dimension_type);
  device_eyes_contours_and_brows->copyToHostTensor(&host_eyes_contours_and_brows);
  device_iris->copyToHostTensor(&host_iris);

  auto iris_dims = host_iris.shape();
  const unsigned int num_iris_element = iris_dims.at(1); // 15=5*3
  const float *iris_ptr = host_iris.host<float>();
  auto eyes_contours_brows_dims = host_eyes_contours_and_brows.shape();
  const unsigned int num_eyes_contours_brows_element = eyes_contours_brows_dims.at(1); // 213=71*3
  const float *eyes_contours_brows_ptr = host_eyes_contours_and_brows.host<float>();

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

#if LITEMNN_DEBUG
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

#if LITEMNN_DEBUG
  std::cout << "detected iris: " << num_iris_element << "\n";
  std::cout << "generate iris landmarks3d num: " << iris.points.size() << "\n";
#endif
}
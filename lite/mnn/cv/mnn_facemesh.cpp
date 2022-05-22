//
// Created by DefTruth on 2022/5/15.
//

#include "mnn_facemesh.h"

using mnncv::MNNFaceMesh;

MNNFaceMesh::MNNFaceMesh(const std::string &_mnn_path, unsigned int _num_threads) :
    mnn_path(_mnn_path.data()), log_id(_mnn_path.data()), num_threads(_num_threads)
{
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // resize tensor & session (NHWC) (1,192,192,3)
  input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
  dimension_type = input_tensor->getDimensionType();
  mnn_interpreter->resizeTensor(
      input_tensor, {input_batch, input_height, input_width, input_channel});
  mnn_interpreter->resizeSession(mnn_session); // may not need
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

MNNFaceMesh::~MNNFaceMesh() noexcept
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNFaceMesh::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (input_tensor) input_tensor->printShape();
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

void MNNFaceMesh::transform(cv::Mat &mat_rs)
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

void MNNFaceMesh::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
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

void MNNFaceMesh::detect(const cv::Mat &mat, types::Landmarks3D &landmarks3d, float &confidence)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat mat_rs;
  FaceMeshScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  this->transform(mat_rs);
  // 2. inference
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. generate landmarks3d and face presence confidence
  this->generate_landmarks3d_and_confidence(
      scale_params, output_tensors, landmarks3d,
      confidence, img_height, img_width);
}

static inline float sigmoid(float x)
{ return (float) (1.f / (1.f + std::exp(-x))); }

void MNNFaceMesh::generate_landmarks3d_and_confidence(
    const FaceMeshScaleParams &scale_params,
    const std::map<std::string, MNN::Tensor *> &output_tensors,
    types::Landmarks3D &landmarks3d, float &confidence,
    int img_height, int img_width)
{
  auto device_landmarks_pred = output_tensors.at("conv2d_21"); // (1,1,1,1404=468*3)
  auto device_conf_pred = output_tensors.at("conv2d_31"); // (1,)
  MNN::Tensor host_landmarks_tensor(device_landmarks_pred, device_landmarks_pred->getDimensionType());
  MNN::Tensor host_conf_tensor(device_conf_pred, device_conf_pred->getDimensionType());
  device_landmarks_pred->copyToHostTensor(&host_landmarks_tensor);
  device_conf_pred->copyToHostTensor(&host_conf_tensor);

  auto output_dims = host_landmarks_tensor.shape();
  const unsigned int num_element = output_dims.at(3); // 1404
  const float *landmarks_ptr = host_landmarks_tensor.host<float>();
  const float *confidence_ptr = host_conf_tensor.host<float>();

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

#if LITEMNN_DEBUG
  std::cout << "detected num_element: " << num_element << "\n";
  std::cout << "generate landmarks3d num: " << landmarks3d.points.size() << "\n";
#endif
}
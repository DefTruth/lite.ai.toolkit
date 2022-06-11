//
// Created by DefTruth on 2022/6/11.
//

#include "mnn_head_seg.h"

using mnncv::MNNHeadSeg;

MNNHeadSeg::MNNHeadSeg(const std::string &_mnn_path, unsigned int _num_threads) :
    mnn_path(_mnn_path.data()), log_id(_mnn_path.data()), num_threads(_num_threads)
{
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // resize tensor & session (NHWC) (1,384,384,3)
  input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
  dimension_type = input_tensor->getDimensionType();
  mnn_interpreter->resizeTensor(
      input_tensor, {input_batch, input_height, input_width, input_channel});
  mnn_interpreter->resizeSession(mnn_session); // may not need
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

MNNHeadSeg::~MNNHeadSeg()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNHeadSeg::print_debug_string()
{
  std::cout << "LITEMNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  input_tensor->printShape();
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

void MNNHeadSeg::transform(const cv::Mat &mat_rs)
{
  cv::Mat canvas;
  cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32FC3, 1.f / 255.f, 0.f);
  MNN::Tensor tmp_host_tensor(input_tensor, input_tensor->getDimensionType());
  std::memcpy(tmp_host_tensor.host<float>(), (void *) canvas.data,
              3 * input_height * input_width * sizeof(float));
  input_tensor->copyFromHostTensor(&tmp_host_tensor); // deep copy
}

void MNNHeadSeg::detect(const cv::Mat &mat, types::HeadSegContent &content)
{
  if (mat.empty()) return;
  const unsigned int img_h = mat.rows;
  const unsigned int img_w = mat.cols;
  const unsigned int channels = mat.channels();
  if (channels != 3) return;
  const unsigned int input_h = input_height; // 384
  const unsigned int input_w = input_width; // 384

  cv::Mat mat_rs;
  cv::resize(mat, mat_rs, cv::Size(input_w, input_h));
  // 1. make input tensor
  this->transform(mat_rs);
  // 2. inference mask (1,384,384,1)
  mnn_interpreter->runSession(mnn_session);
  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. post process.
  auto device_mask_pred = output_tensors.at("sigmoid/Sigmoid:0");
  MNN::Tensor host_mask_tensor(device_mask_pred, device_mask_pred->getDimensionType());
  device_mask_pred->copyToHostTensor(&host_mask_tensor);

  auto mask_dims = host_mask_tensor.shape();
  const unsigned int out_h = mask_dims.at(1); // 384
  const unsigned int out_w = mask_dims.at(2); // 384
  float *mask_ptr = host_mask_tensor.host<float>();

  cv::Mat mask_adj;
  cv::Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr);
  cv::resize(mask_out, mask_adj, cv::Size(img_w, img_h)); // (img_h,img_w,1)

  content.mask = mask_adj;
  content.flag = true;
}

//
// Created by DefTruth on 2021/11/29.
//

#include "mnn_deeplabv3_resnet101.h"
#include "lite/utils.h"

using mnncv::MNNDeepLabV3ResNet101;

MNNDeepLabV3ResNet101::MNNDeepLabV3ResNet101(
    const std::string &_mnn_path, unsigned int _num_threads
) : log_id(_mnn_path.data()),
    mnn_path(_mnn_path.data()),
    num_threads(_num_threads)
{
  initialize_interpreter();
  initialize_pretreat();
}

MNNDeepLabV3ResNet101::~MNNDeepLabV3ResNet101()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
}

void MNNDeepLabV3ResNet101::initialize_interpreter()
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
  input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
  // 5. init input dims
  dynamic_input_height = input_tensor->height();
  dynamic_input_width = input_tensor->width();
  dimension_type = input_tensor->getDimensionType(); // CAFFE(NCHW)
  mnn_interpreter->resizeTensor(input_tensor, {1, 3, dynamic_input_height, dynamic_input_width});
  mnn_interpreter->resizeSession(mnn_session);
#ifdef LITEMNN_DEBUG
  this->print_debug_string();
#endif
}

void MNNDeepLabV3ResNet101::initialize_pretreat()
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

void MNNDeepLabV3ResNet101::transform(const cv::Mat &mat)
{
  const int img_width = mat.cols;
  const int img_height = mat.rows;
  // update dynamic input dims
  dynamic_input_height = img_height;
  dynamic_input_width = img_width;

  // update input tensor and resize Session
  mnn_interpreter->resizeTensor(input_tensor, {1, 3, dynamic_input_height, dynamic_input_width});
  mnn_interpreter->resizeSession(mnn_session);

  // push data into input tensor
  pretreat->convert(mat.data, dynamic_input_width, dynamic_input_height, mat.step[0], input_tensor);
}

void MNNDeepLabV3ResNet101::detect(const cv::Mat &mat, types::SegmentContent &content)
{
  if (mat.empty()) return;
  // 1. make input tensor
  this->transform(mat);
  // 2. inference & run session
  mnn_interpreter->runSession(mnn_session);

  auto output_tensors = mnn_interpreter->getSessionOutputAll(mnn_session);
  // 3. fetch
  auto device_scores_ptr = output_tensors.at("out"); // (1,21,h,w)
  MNN::Tensor host_scores_tensor(device_scores_ptr, device_scores_ptr->getDimensionType());
  device_scores_ptr->copyToHostTensor(&host_scores_tensor);
#ifdef LITEMNN_DEBUG
  host_scores_tensor.printShape();
#endif

  auto scores_dims = host_scores_tensor.shape();
  const unsigned int output_classes = scores_dims.at(1);
  const unsigned int output_height = scores_dims.at(2);
  const unsigned int output_width = scores_dims.at(3);

  const float *scores_ptr = host_scores_tensor.host<float>();
  // time cost!
  content.names_map.clear();
  content.class_mat = cv::Mat(output_height, output_width, CV_8UC1, cv::Scalar(0));
  content.color_mat = mat.clone();

  const unsigned int scores_step = output_height * output_width; // h x w

  for (unsigned int i = 0; i < output_height; ++i)
  {

    uchar *p_class = content.class_mat.ptr<uchar>(i);
    cv::Vec3b *p_color = content.color_mat.ptr<cv::Vec3b>(i);

    for (unsigned int j = 0; j < output_width; ++j)
    {
      // argmax
      unsigned int max_label = 0;
      float max_conf = scores_ptr[0 * scores_step + i * output_width + j];

      for (unsigned int l = 0; l < output_classes; ++l)
      {
        float conf = scores_ptr[l * scores_step + i * output_width + j];
        if (conf > max_conf)
        {
          max_conf = conf;
          max_label = l;
        }
      }

      if (max_label == 0) continue;

      // assign label for pixel(i,j)
      p_class[j] = cv::saturate_cast<uchar>(max_label);
      // assign color for detected class at pixel(i,j).
      p_color[j][0] = cv::saturate_cast<uchar>((max_label % 10) * 20);
      p_color[j][1] = cv::saturate_cast<uchar>((max_label % 5) * 40);
      p_color[j][2] = cv::saturate_cast<uchar>((max_label % 10) * 20);
      // assign names map
      content.names_map[max_label] = class_names[max_label - 1]; // max_label >= 1
    }

  }

  content.flag = true;
}

void MNNDeepLabV3ResNet101::print_debug_string()
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
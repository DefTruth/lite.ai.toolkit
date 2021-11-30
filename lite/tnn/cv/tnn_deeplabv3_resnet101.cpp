//
// Created by DefTruth on 2021/11/29.
//

#include "tnn_deeplabv3_resnet101.h"
#include "lite/utils.h"

using tnncv::TNNDeepLabV3ResNet101;

TNNDeepLabV3ResNet101::TNNDeepLabV3ResNet101(
    const std::string &_proto_path,
    const std::string &_model_path,
    unsigned int _num_threads
) : proto_path(_proto_path.data()),
    model_path(_model_path.data()),
    log_id(_proto_path.data()),
    num_threads(_num_threads)
{
  initialize_instance();
}

TNNDeepLabV3ResNet101::~TNNDeepLabV3ResNet101()
{
  net = nullptr;
  input_mat = nullptr;
  instance = nullptr;
}

void TNNDeepLabV3ResNet101::initialize_instance()
{
  std::string proto_content_buffer, model_content_buffer;
  proto_content_buffer = BasicTNNHandler::content_buffer_from(proto_path);
  model_content_buffer = BasicTNNHandler::content_buffer_from(model_path);

  tnn::ModelConfig model_config;
  model_config.model_type = tnn::MODEL_TYPE_TNN;
  model_config.params = {proto_content_buffer, model_content_buffer};

  // 1. init TNN net
  tnn::Status status;
  net = std::make_shared<tnn::TNN>();
  status = net->Init(model_config);
  if (status != tnn::TNN_OK || !net)
  {
#ifdef LITETNN_DEBUG
    std::cout << "net->Init failed!\n";
#endif
    return;
  }
  // 2. init device type, change this default setting
  // for better performance. such as CUDA/OPENCL/...
#ifdef __ANDROID__
  network_device_type = tnn::DEVICE_ARM; // CPU,GPU
  input_device_type = tnn::DEVICE_ARM; // CPU only
  output_device_type = tnn::DEVICE_ARM;
#else
  network_device_type = tnn::DEVICE_X86; // CPU,GPU
  input_device_type = tnn::DEVICE_X86; // CPU only
  output_device_type = tnn::DEVICE_X86;
#endif
  // 3. init instance
  tnn::NetworkConfig network_config;
  network_config.library_path = {""};
  network_config.device_type = network_device_type;

  instance = net->CreateInst(network_config, status);
  if (status != tnn::TNN_OK || !instance)
  {
#ifdef LITETNN_DEBUG
    std::cout << "CreateInst failed!" << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. setting up num_threads
  instance->SetCpuNumThreads((int) num_threads);
  // 5. init input information.
  input_shape = BasicTNNHandler::get_input_shape(instance, "input");

  if (input_shape.size() != 4)
  {
#ifdef LITETNN_DEBUG
    throw std::runtime_error("Found input_shape.size()!=4, but "
                             "input only support 4 dims."
                             "Such as NCHW, NHWC ...");
#else
    return;
#endif
  }
  input_mat_type = BasicTNNHandler::get_input_mat_type(instance, "input");
  input_data_format = BasicTNNHandler::get_input_data_format(instance, "input");
  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    dynamic_input_height = input_shape.at(2);
    dynamic_input_width = input_shape.at(3);
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    dynamic_input_height = input_shape.at(1);
    dynamic_input_width = input_shape.at(2);
  } // unsupport
  else
  {
#ifdef LITETNN_DEBUG
    std::cout << "input only support NCHW and NHWC "
                 "input_data_format, but found others.\n";
#endif
    return;
  }
  // 6. init output information, debug only.
  output_shape = BasicTNNHandler::get_output_shape(instance, "out");
#ifdef LITETNN_DEBUG
  this->print_debug_string();
#endif
}

void TNNDeepLabV3ResNet101::print_debug_string()
{
  std::cout << "LITETNN_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  BasicTNNHandler::print_name_shape("input", input_shape);
  std::string data_format_string =
      (input_data_format == tnn::DATA_FORMAT_NCHW) ? "NCHW" : "NHWC";
  std::cout << "Input Data Format: " << data_format_string << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  BasicTNNHandler::print_name_shape("out", output_shape);
  std::cout << "========================================\n";
}

void TNNDeepLabV3ResNet101::transform(const cv::Mat &mat)
{
  const int img_width = mat.cols;
  const int img_height = mat.rows;
  // update dynamic input dims
  dynamic_input_height = img_height;
  dynamic_input_width = img_width;
  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    input_shape.at(2) = dynamic_input_height;
    input_shape.at(3) = dynamic_input_width;
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    input_shape.at(1) = dynamic_input_height;
    input_shape.at(2) = dynamic_input_width;
  }

  // update input mat and reshape instance
  // reference: https://github.com/Tencent/TNN/blob/master/examples/base/ocr_text_recognizer.cc#L120
  tnn::InputShapesMap input_shape_map;
  input_shape_map.insert({"input", input_shape});

  auto status = instance->Reshape(input_shape_map);
  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << "instance Reshape failed in TNNDeepLabV3ResNet101\n";
#endif
  }

  cv::Mat canvas;
  cv::cvtColor(mat, canvas, cv::COLOR_BGR2RGB);
  // push into input_mat
  input_mat = std::make_shared<tnn::Mat>(
      input_device_type,
      tnn::N8UC3,
      input_shape,
      (void *) canvas.data
  );
  if (!input_mat->GetData())
  {
#ifdef LITETNN_DEBUG
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void TNNDeepLabV3ResNet101::detect(const cv::Mat &mat, types::SegmentContent &content)
{
  if (mat.empty()) return;

  // 1. make input mat
  this->transform(mat);
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
  // 4. fetch
  tnn::MatConvertParam cvt_param;
  std::shared_ptr<tnn::Mat> scores_mat; // (1,21,h,w)
  status = instance->GetOutputMat(scores_mat, cvt_param, "out", output_device_type);

  if (status != tnn::TNN_OK)
  {
#ifdef LITETNN_DEBUG
    std::cout << status.description().c_str() << "\n";
#endif
    return;
  }

  auto scores_dims = scores_mat->GetDims();
#ifdef LITETNN_DEBUG
  BasicTNNHandler::print_name_shape("out", scores_dims);
#endif

  const unsigned int output_classes = scores_dims.at(1);
  const unsigned int output_height = scores_dims.at(2);
  const unsigned int output_width = scores_dims.at(3);

  const float *scores_ptr = (float *) scores_mat->GetData();

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



















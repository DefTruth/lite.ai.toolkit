# Rapid implementation of your inference using BasicOrtHandler
For the single input and multiple outputs model, you can inherit [BasicOrthHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h) and then implement the
`transform` and `detect` interfaces, it's just a simple wrapper of the onnxruntime c++ inference context. Or you can implement a `xxx_model.cpp` completely on your own. 
After submitting MR, I will modify it in the same way that I inherited `BasicOrtHandler` and add it to the library. For the multiple inputs and multiple outputs model, maybe you can try to inherit [BasicMultiOrthHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h).   
For example:
* inherit [BasicOrthHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h).
```c++
#include "ort/core/ort_core.h"

namespace ortcv
{
    class AgeGoogleNet : public BasicOrtHandler
    {
    private:
      const float mean_val[3] = {104.0f, 117.0f, 123.0f};
      const float scale_val[3] = {1.0f, 1.0f, 1.0f};
      const unsigned int age_intervals[8][2] = {
            {0,  2},
            {4,  6},
            {8,  12},
            {15, 20},
            {25, 32},
            {38, 43},
            {48, 53},
            {60, 100}
        };
    
    public:
      explicit AgeGoogleNet(const std::string &_onnx_path, unsigned int _num_threads = 1) :
      BasicOrtHandler(_onnx_path, _num_threads)
      {};
        
      ~AgeGoogleNet()
      {};
        
    private:
        ort::Value transform(const cv::Mat &mat);
    
    public:
      void detect(const cv::Mat &mat, types::Age &age);
};

}
``` 
* implementations for `transform` and `detect` interfaces.
```c++
#include "age_googlenet.h"
#include "ort/core/ort_utils.h"

using ortcv::AgeGoogleNet;

ort::Value AgeGoogleNet::transform(const cv::Mat &mat)
{
  cv::Mat canva = mat.clone();
  cv::resize(canva, canva, cv::Size(input_node_dims.at(3), input_node_dims.at(2)));
  cv::cvtColor(canva, canva, cv::COLOR_BGR2RGB);
  // (1,3,224,224)
  ortcv::utils::transform::normalize_inplace(canva, mean_val, scale_val); // float32
    
  return ortcv::utils::transform::mat3f_to_tensor(
    canva, input_node_dims, memory_info_handler,
    input_values_handler, ortcv::utils::transform::CHW);
}

void AgeGoogleNet::detect(const cv::Mat &mat, types::Age &age)
{
  if (mat.empty()) return;
  // 1. make input tensor
  ort::Value input_tensor = this->transform(mat);
  // 2. inference
  auto output_tensors = ort_session->Run(
    ort::RunOptions{nullptr}, input_node_names.data(),
    &input_tensor, 1, output_node_names.data(), num_outputs
    );
  ort::Value &age_logits = output_tensors.at(0); // (1,8)
  auto age_dims = output_node_dims.at(0);
  unsigned int interval = 0;
  const unsigned int num_intervals = age_dims.at(1); // 8
  const float *pred_logits = age_logits.GetTensorMutableData<float>();
  auto softmax_probs = ortcv::utils::math::softmax<float>(pred_logits, num_intervals, interval);
  const float pred_age = static_cast<float>(age_intervals[interval][0] + age_intervals[interval][1]) / 2.0f;
  age.age = pred_age;
  age.age_interval[0] = age_intervals[interval][0];
  age.age_interval[1] = age_intervals[interval][1];
  age.interval_prob = softmax_probs[interval];
  age.flag = true;
}
```
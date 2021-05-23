# Rapid implementation of your inference using BasicOrtHandler

对于单输入多输出的模型，你可以考虑继承 [BasicOrthHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h) 类，然后实现`transform`和`detect`方法，`transform`方法是虚函数，而`detect`方法不是必要的，你完全可以实现另一个不同名但功能类似的方法。`transform`方法应负责将格式类似`cv::Mat`的数据转换成`Ort::Value`张量，注意，在我的代码中，为了方便，我使用了`namespace ort=Ort`. 调用`litehub/ort/core/ort_utils.h`中的`ortcv::utils::create_tensor`方法，可以很方便地将`cv::Mat`转换为`CHW`或`HWC(未测试)`的`Ort::Value`张量。事实上`	BasicOrtHandler`只是onnxruntime的c++接口在使用过程中所涉及上下文的一个简单封装，看了`BasicOrtHandler`的实现后，你会发现并不复杂。对于多输入多输出模型，你可以参考[BasicMultiOrtHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h). 具体使用方法如下：

* 继承 [BasicOrthHandler](https://github.com/DefTruth/litehub/blob/main/ort/core/ort_handler.h) 

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
        
      ~AgeGoogleNet() override = default;
        
    private:
        ort::Value transform(const cv::Mat &mat) override;
    
    public:
      void detect(const cv::Mat &mat, types::Age &age);
    };
}
```

* 实现 `transform` 和 `detect` 方法.

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
    
  return ortcv::utils::transform::create_tensor(
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




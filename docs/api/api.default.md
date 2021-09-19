# Default Version APIs.

More details of basic types for Default Version APIs can be found at [types](https://github.com/DefTruth/lite.ai.toolkit/blob/main/ort/core/ort_types.h) . Note that Lite.AI.ToolKit uses `onnxruntime` as default backend, for the reason that onnxruntime supports the most of onnx's operators. `(TODO: Add detailed API documentation)`


> `lite::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```  

> `lite::cv::detection::YoloV4`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::cv::detection::YoloV3`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);
```

> `lite::cv::detection::TinyYoloV3`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);
```

> `lite::cv::detection::SSD`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::cv::detection::SSDMobileNetV1`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::cv::face::FSANet`
```c++
void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
```

> `lite::cv::face::UltraFace`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
            float score_threshold = 0.7f, float iou_threshold = 0.3f,
            unsigned int topk = 300, unsigned int nms_type = 0);
```

> `lite::cv::face::PFLD`
```c++
void detect(const cv::Mat &mat, types::Landmarks &landmarks);
```  

> `lite::cv::face::AgeGoogleNet`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```  

> `lite::cv::face::GenderGoogleNet`
```c++
void detect(const cv::Mat &mat, types::Gender &gender);
```

> `lite::cv::face::VGG16Age`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```

> `lite::cv::face::VGG16Gender`
```c++
void detect(const cv::Mat &mat, types::Gender &gender);
```  

> `lite::cv::face::EmotionFerPlus`
```c++
void detect(const cv::Mat &mat, types::Emotions &emotions);
```

> `lite::cv::face::SSRNet`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```

> `lite::cv::faceid::ArcFaceResNet`
```c++
void detect(const cv::Mat &mat, types::FaceContent &face_content);
```  

> `lite::cv::segmentation::DeepLabV3ResNet101`
```c++
void detect(const cv::Mat &mat, types::SegmentContent &content);
```  

> `lite::cv::segmentation::FCNResNet101`
```c++
void detect(const cv::Mat &mat, types::SegmentContent &content);
```  

> `lite::cv::style::FastStyleTransfer`
```c++
void detect(const cv::Mat &mat, types::StyleContent &style_content);
```

> `lite::cv::colorization::Colorizer`
```c++
void detect(const cv::Mat &mat, types::ColorizeContent &colorize_content);
```

> `lite::cv::resolution::SubPixelCNN`
```c++
void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
```

> `lite::cv::classification::EfficientNetLite4`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```

> `lite::cv::classification::ShuffleNetV2`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::DenseNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::GhostNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::HdrDNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::MobileNetV2`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::ResNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::classification::ResNeXt`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::cv::utils::hard_nms`
```c++
LITEHUB_EXPORTS void
hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
```

> `lite::cv::utils::blending_nms`
```c++
LITEHUB_EXPORTS void
blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
```

> `lite::cv::utils::offset_nms`
```c++
LITEHUB_EXPORTS void
offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
```

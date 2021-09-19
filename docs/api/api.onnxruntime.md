# ONNXRuntime Version APIs.

More details of basic types for ONNXRuntime Version APIs can be found at [ort_types](https://github.com/DefTruth/lite.ai.toolkit/blob/main/ort/core/ort_types.h) . `(TODO: Add detailed API documentation).`

> `lite::onnxruntime::cv::detection::YoloV5`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::onnxruntime::cv::detection::YoloV4`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::onnxruntime::cv::detection::YoloV3`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);
```

> `lite::onnxruntime::cv::detection::TinyYoloV3`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes);
```

> `lite::onnxruntime::cv::detection::SSD`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::onnxruntime::cv::detection::SSDMobileNetV1`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes, 
            float score_threshold = 0.25f, float iou_threshold = 0.45f,
            unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
```

> `lite::onnxruntime::cv::face::FSANet`
```c++
void detect(const cv::Mat &mat, types::EulerAngles &euler_angles);
```

> `lite::onnxruntime::cv::face::UltraFace`
```c++
void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
            float score_threshold = 0.7f, float iou_threshold = 0.3f,
            unsigned int topk = 300, unsigned int nms_type = 0);
```

> `lite::onnxruntime::cv::face::PFLD`
```c++
void detect(const cv::Mat &mat, types::Landmarks &landmarks);
```  

> `lite::onnxruntime::cv::face::AgeGoogleNet`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```

> `lite::onnxruntime::cv::face::GenderGoogleNet`
```c++
void detect(const cv::Mat &mat, types::Gender &gender);
```

> `lite::onnxruntime::cv::face::VGG16Age`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```

> `lite::onnxruntime::cv::face::VGG16Gender`
```c++
void detect(const cv::Mat &mat, types::Gender &gender);
```  

> `lite::onnxruntime::cv::face::EmotionFerPlus`
```c++
void detect(const cv::Mat &mat, types::Emotions &emotions);
```

> `lite::onnxruntime::cv::face::SSRNet`
```c++
void detect(const cv::Mat &mat, types::Age &age);
```

> `lite::onnxruntime::cv::faceid::ArcFaceResNet`
```c++
void detect(const cv::Mat &mat, types::FaceContent &face_content);
```

> `lite::onnxruntime::cv::segmentation::DeepLabV3ResNet101`
```c++
void detect(const cv::Mat &mat, types::SegmentContent &content);
```

> `lite::onnxruntime::cv::segmentation::FCNResNet101`
```c++
void detect(const cv::Mat &mat, types::SegmentContent &content);
```  

> `lite::onnxruntime::cv::style::FastStyleTransfer`
```c++
void detect(const cv::Mat &mat, types::StyleContent &style_content);
```

> `lite::onnxruntime::cv::colorization::Colorizer`
```c++
void detect(const cv::Mat &mat, types::ColorizeContent &colorize_content);
```

> `lite::onnxruntime::cv::resolution::SubPixelCNN`
```c++
void detect(const cv::Mat &mat, types::SuperResolutionContent &super_resolution_content);
```  

> `lite::onnxruntime::cv::classification::EfficientNetLite4`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```

> `lite::onnxruntime::cv::classification::ShuffleNetV2`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::DenseNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::GhostNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::HdrDNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::MobileNetV2`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::ResNet`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```  

> `lite::onnxruntime::cv::classification::ResNeXt`
```c++
void detect(const cv::Mat &mat, types::ImageNetContent &content, unsigned int top_k = 5);
```
# Lite.AI.ToolKit.Hub.ONNX 

Correspondence between the classes in *Lite.AI.ToolKit* and pretrained model files can be found at this document. For examples, the pretrained model files for *lite::cv::detection::YoloV5* and *lite::cv::detection::YoloX* are listed as follows.

|             Class             | Pretrained ONNX Files |                 Rename or Converted From (Repo)                  | Size  |
|:-----------------------------:|:---------------------:|:----------------------------------------------------------------:|:-----:|
| *lite::cv::detection::YoloV5* |     yolov5l.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 188Mb |
| *lite::cv::detection::YoloV5* |     yolov5m.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 85Mb  |
| *lite::cv::detection::YoloV5* |     yolov5s.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 29Mb  |
| *lite::cv::detection::YoloV5* |     yolov5x.onnx      |    [yolov5](https://github.com/ultralytics/yolov5) (🔥🔥💥↑)     | 351Mb |
| *lite::cv::detection::YoloX*  |     yolox_x.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 378Mb |
| *lite::cv::detection::YoloX*  |     yolox_l.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 207Mb |
| *lite::cv::detection::YoloX*  |     yolox_m.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 97Mb  |
| *lite::cv::detection::YoloX*  |     yolox_s.onnx      | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 34Mb  |
| *lite::cv::detection::YoloX*  |    yolox_tiny.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 19Mb  |
| *lite::cv::detection::YoloX*  |    yolox_nano.onnx    | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (🔥🔥!!↑) | 3.5Mb |

It means that you can load the any one `yolov5*.onnx` and  `yolox_*.onnx` according to your application through the same Lite.AI classes, such as *YoloV5*, *YoloX*, etc.

```c++
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5x.onnx");  // for server
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5l.onnx"); 
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5m.onnx");  
auto *yolov5 = new lite::cv::detection::YoloV5("yolov5s.onnx");  // for mobile device 
auto *yolox = new lite::cv::detection::YoloX("yolox_x.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_l.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_m.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_s.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_tiny.onnx");  
auto *yolox = new lite::cv::detection::YoloX("yolox_nano.onnx");  // 3.5Mb only !
```

You can download all the pretrained models files of ONNX format from ([Baidu Drive](https://pan.baidu.com/s/1elUGcx7CZkkjEoYhTMwTRQ) code: 8gin) 

## Object Detection.  

<div id="lite.ai.toolkit.hub.onnx-object-detection"></div>

|                     Class                      |         Pretrained ONNX Files          |                          Rename or Converted From (Repo)                          | Size  |
|:----------------------------------------------:|:--------------------------------------:|:---------------------------------------------------------------------------------:|:-----:|
|         *lite::cv::detection::YoloV5*          |              yolov5l.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 188Mb |
|         *lite::cv::detection::YoloV5*          |              yolov5m.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 85Mb  |
|         *lite::cv::detection::YoloV5*          |              yolov5s.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 29Mb  |
|         *lite::cv::detection::YoloV5*          |              yolov5x.onnx              |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 351Mb |
|          *lite::cv::detection::YoloX*          |              yolox_x.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 378Mb |
|          *lite::cv::detection::YoloX*          |              yolox_l.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 207Mb |
|          *lite::cv::detection::YoloX*          |              yolox_m.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 97Mb  |
|          *lite::cv::detection::YoloX*          |              yolox_s.onnx              |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 34Mb  |
|          *lite::cv::detection::YoloX*          |            yolox_tiny.onnx             |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 19Mb  |
|          *lite::cv::detection::YoloX*          |            yolox_nano.onnx             |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 3.5Mb |
|         *lite::cv::detection::YoloV3*          |             yolov3-10.onnx             |                   [onnx-models](https://github.com/onnx/models)                   | 236Mb |
|       *lite::cv::detection::TinyYoloV3*        |          tiny-yolov3-11.onnx           |                   [onnx-models](https://github.com/onnx/models)                   | 33Mb  |
|         *lite::cv::detection::YoloV4*          |    voc-mobilenetv2-yolov4-640.onnx     |             [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch)             | 176Mb |
|         *lite::cv::detection::YoloV4*          |    voc-mobilenetv2-yolov4-416.onnx     |             [YOLOv4...](https://github.com/argusswift/YOLOv4-pytorch)             | 176Mb |
|           *lite::cv::detection::SSD*           |              ssd-10.onnx               |                   [onnx-models](https://github.com/onnx/models)                   | 76Mb  |
|          *lite::cv::detection::YoloR*          |        yolor-d6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 667Mb |
|          *lite::cv::detection::YoloR*          |         yolor-d6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 601Mb |
|          *lite::cv::detection::YoloR*          |         yolor-d6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 584Mb |
|          *lite::cv::detection::YoloR*          |        yolor-e6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 530Mb |
|          *lite::cv::detection::YoloR*          |         yolor-e6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 464Mb |
|          *lite::cv::detection::YoloR*          |         yolor-e6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 448Mb |
|          *lite::cv::detection::YoloR*          |        yolor-p6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 214Mb |
|          *lite::cv::detection::YoloR*          |         yolor-p6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 160Mb |
|          *lite::cv::detection::YoloR*          |         yolor-p6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 147Mb |
|          *lite::cv::detection::YoloR*          |        yolor-w6-1280-1280.onnx         |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 382Mb |
|          *lite::cv::detection::YoloR*          |         yolor-w6-640-640.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 324Mb |
|          *lite::cv::detection::YoloR*          |         yolor-w6-320-320.onnx          |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 309Mb |
|          *lite::cv::detection::YoloR*          |     yolor-ssss-s2d-1280-1280.onnx      |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 90Mb  |
|          *lite::cv::detection::YoloR*          |      yolor-ssss-s2d-640-640.onnx       |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 49Mb  |
|          *lite::cv::detection::YoloR*          |      yolor-ssss-s2d-320-320.onnx       |                   [yolor](https://github.com/WongKinYiu/yolor)                    | 39Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |      yolov4_tiny_weights_voc.onnx      |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |    yolov4_tiny_weights_voc_SE.onnx     |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |   yolov4_tiny_weights_voc_CBAM.onnx    |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::TinyYoloV4VOC*      |    yolov4_tiny_weights_voc_ECA.onnx    |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|     *lite::cv::detection::TinyYoloV4COCO*      |     yolov4_tiny_weights_coco.onnx      |       [yolov4-tiny...](https://github.com/bubbliiiing/yolov4-tiny-pytorch)        | 23Mb  |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p5-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5-320-320.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p6-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6-320-320.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p7-1280-1280.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 1.1Gb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p7-640-640.onnx   |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 1.1Gb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p5_-1280-1280.onnx |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5_-640-640.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p5_-320-320.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 270Mb |
|      *lite::cv::detection::ScaledYoloV4*       | ScaledYoloV4_yolov4-p6_-1280-1280.onnx |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6_-640-640.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::ScaledYoloV4*       |  ScaledYoloV4_yolov4-p6_-320-320.onnx  |            [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)             | 487Mb |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d0.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 15Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d1.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 26Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d2.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 32Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d3.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 49Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d4.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 85Mb  |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d5.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 138Mb |
|      *lite::cv::detection::EfficientDet*       |          efficientdet-d6.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 220Mb |
|     *lite::cv::detection::EfficientDetD7*      |          efficientdet-d7.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 220Mb |
|     *lite::cv::detection::EfficientDetD8*      |          efficientdet-d8.onnx          | [...EfficientDet...](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) | 322Mb |
|          *lite::cv::detection::YOLOP*          |          yolop-1280-1280.onnx          |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|          *lite::cv::detection::YOLOP*          |           yolop-640-640.onnx           |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|          *lite::cv::detection::YOLOP*          |           yolop-320-320.onnx           |                     [YOLOP](https://github.com/hustvl/YOLOP)                      | 30Mb  |
|         *lite::cv::detection::NanoDet*         |          nanodet_m_0.5x.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 1.1Mb |
|         *lite::cv::detection::NanoDet*         |             nanodet_m.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 3.6Mb |
|         *lite::cv::detection::NanoDet*         |          nanodet_m_1.5x.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 7.9Mb |
|         *lite::cv::detection::NanoDet*         |        nanodet_m_1.5x_416.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 7.9Mb |
|         *lite::cv::detection::NanoDet*         |           nanodet_m_416.onnx           |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 3.6Mb |
|         *lite::cv::detection::NanoDet*         |             nanodet_g.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 14Mb  |
|         *lite::cv::detection::NanoDet*         |             nanodet_t.onnx             |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 5.1Mb |
|         *lite::cv::detection::NanoDet*         |       nanodet-RepVGG-A0_416.onnx       |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 26Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite0_320.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 12Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite1_416.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 15Mb  |
| *lite::cv::detection::NanoDetEfficientNetLite* |  nanodet-EfficientNet-Lite2_512.onnx   |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 18Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_x_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 378Mb |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_l_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 207Mb |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_m_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 97Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |          yolox_s_v0.1.1.onnx           |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 34Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |         yolox_tiny_v0.1.1.onnx         |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 19Mb  |
|      *lite::cv::detection::YoloX_V_0_1_1*      |         yolox_nano_v0.1.1.onnx         |              [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)               | 3.5Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5l.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 178Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5m.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 81Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5s.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 28Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5x.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 331Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |       yolov5n.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 7.5Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5l6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 294Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5m6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 128Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5s6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 50Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5x6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 538Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |      yolov5n6.640-640.v.6.0.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 14Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5l6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 294Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5m6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 128Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5s6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 50Mb  |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5x6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 538Mb |
|      *lite::cv::detection::YoloV5_V_6_0*       |     yolov5n6.1280-1280.v.6.0.onnx      |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 14Mb  |
|       *lite::cv::detection::NanoDetPlus*       |        nanodet-plus-m_320.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 4.5Mb |
|       *lite::cv::detection::NanoDetPlus*       |        nanodet-plus-m_416.onnx         |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 4.5Mb |
|       *lite::cv::detection::NanoDetPlus*       |      nanodet-plus-m-1.5x_320.onnx      |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 9.4Mb |
|       *lite::cv::detection::NanoDetPlus*       |      nanodet-plus-m-1.5x_416.onnx      |                  [nanodet](https://github.com/RangiLyu/nanodet)                   | 9.4Mb |
|        *lite::cv::detection::InsectDet*        |     quarrying_insect_detector.onnx     |           [InsectID](https://github.com/quarrying/quarrying-insect-id)            | 22Mb  |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5l.v6.1.640x640.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 178Mb |
|      *lite::cv::detection::YoloV5_V_6_1*       |      yolov5l.v6.1.1280x1280.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 178Mb |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5m.v6.1.640x640.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 81Mb  |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5x.v6.1.640x640.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 332Mb |
|      *lite::cv::detection::YoloV5_V_6_1*       |      yolov5x.v6.1.1280x1280.onnx       |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 332Mb |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5s.v6.1.640x640.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 28Mb  |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5s.v6.1.320x320.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  | 28Mb  |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5n.v6.1.640x640.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  |  7Mb  |
|      *lite::cv::detection::YoloV5_V_6_1*       |       yolov5n.v6.1.320x320.onnx        |                  [yolov5](https://github.com/ultralytics/yolov5)                  |  7Mb  |

## Classification.  

<div id="lite.ai.toolkit.hub.onnx-classification"></div>


|                    Class                     |      Pretrained ONNX Files       |               Rename or Converted From (Repo)                | Size  |
|:--------------------------------------------:|:--------------------------------:|:------------------------------------------------------------:|:-----:|
| *lite::cv::classification:EfficientNetLite4* |    efficientnet-lite4-11.onnx    |        [onnx-models](https://github.com/onnx/models)         | 49Mb  |
|   *lite::cv::classification::ShuffleNetV2*   |      shufflenet-v2-10.onnx       |        [onnx-models](https://github.com/onnx/models)         | 8.7Mb |
|   *lite::cv::classification::DenseNet121*    |         densenet121.onnx         |       [torchvision](https://github.com/pytorch/vision)       | 30Mb  |
|     *lite::cv::classification::GhostNet*     |          ghostnet.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 20Mb  |
|     *lite::cv::classification::HdrDNet*      |           hardnet.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 13Mb  |
|      *lite::cv::classification::IBNNet*      |          ibnnet18.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 97Mb  |
|   *lite::cv::classification::MobileNetV2*    |         mobilenetv2.onnx         |       [torchvision](https://github.com/pytorch/vision)       | 13Mb  |
|      *lite::cv::classification::ResNet*      |          resnet18.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 44Mb  |
|     *lite::cv::classification::ResNeXt*      |           resnext.onnx           |       [torchvision](https://github.com/pytorch/vision)       | 95Mb  |
|     *lite::cv::classification::InsectID*     | quarrying_insect_identifier.onnx | [InsectID](https://github.com/quarrying/quarrying-insect-id) | 27Mb  |
|      *lite::cv::classification:PlantID*      |   quarrying_plantid_model.onnx   |  [PlantID](https://github.com/quarrying/quarrying-plant-id)  | 30Mb  |


## Face Detection.

<div id="lite.ai.toolkit.hub.onnx-face-detection"></div>  

|                   Class                   |            Pretrained ONNX Files            |                             Rename or Converted From (Repo)                             |  Size  |
|:-----------------------------------------:|:-------------------------------------------:|:---------------------------------------------------------------------------------------:|:------:|
|    *lite::cv::face::detect::UltraFace*    |           ultraface-rfb-640.onnx            | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.5Mb  |
|    *lite::cv::face::detect::UltraFace*    |           ultraface-rfb-320.onnx            | [Ultra-Light...](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) | 1.2Mb  |
|   *lite::cv::face::detect::RetinaFace*    |      Pytorch_RetinaFace_resnet50.onnx       |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
|   *lite::cv::face::detect::RetinaFace*    |  Pytorch_RetinaFace_resnet50-640-640.onnx   |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
|   *lite::cv::face::detect::RetinaFace*    |  Pytorch_RetinaFace_resnet50-320-320.onnx   |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
|   *lite::cv::face::detect::RetinaFace*    |  Pytorch_RetinaFace_resnet50-720-1080.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 104Mb  |
|   *lite::cv::face::detect::RetinaFace*    |     Pytorch_RetinaFace_mobile0.25.onnx      |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
|   *lite::cv::face::detect::RetinaFace*    | Pytorch_RetinaFace_mobile0.25-640-640.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
|   *lite::cv::face::detect::RetinaFace*    | Pytorch_RetinaFace_mobile0.25-320-320.onnx  |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
|   *lite::cv::face::detect::RetinaFace*    | Pytorch_RetinaFace_mobile0.25-720-1080.onnx |             [...Retinaface](https://github.com/biubug6/Pytorch_Retinaface)              | 1.6Mb  |
|    *lite::cv::face::detect::FaceBoxes*    |               FaceBoxes.onnx                |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|    *lite::cv::face::detect::FaceBoxes*    |           FaceBoxes-640-640.onnx            |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|    *lite::cv::face::detect::FaceBoxes*    |           FaceBoxes-320-320.onnx            |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|    *lite::cv::face::detect::FaceBoxes*    |           FaceBoxes-720-1080.onnx           |                [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)                | 3.8Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_500m_shape160x160.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_500m_shape320x320.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_500m_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |
|      *lite::cv::face::detect::SCRFD*      |     scrfd_500m_bnkps_shape160x160.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|      *lite::cv::face::detect::SCRFD*      |     scrfd_500m_bnkps_shape320x320.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|      *lite::cv::face::detect::SCRFD*      |     scrfd_500m_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.5Mb  |  
|      *lite::cv::face::detect::SCRFD*      |         scrfd_1g_shape160x160.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|      *lite::cv::face::detect::SCRFD*      |         scrfd_1g_shape320x320.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|      *lite::cv::face::detect::SCRFD*      |         scrfd_1g_shape640x640.onnx          |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 2.7Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_2.5g_shape160x160.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_2.5g_shape320x320.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_2.5g_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |
|      *lite::cv::face::detect::SCRFD*      |     scrfd_2.5g_bnkps_shape160x160.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|      *lite::cv::face::detect::SCRFD*      |     scrfd_2.5g_bnkps_shape320x320.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|      *lite::cv::face::detect::SCRFD*      |     scrfd_2.5g_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 3.3Mb  |  
|      *lite::cv::face::detect::SCRFD*      |         scrfd_10g_shape640x640.onnx         |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |
|      *lite::cv::face::detect::SCRFD*      |        scrfd_10g_shape1280x1280.onnx        |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |
|      *lite::cv::face::detect::SCRFD*      |      scrfd_10g_bnkps_shape640x640.onnx      |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |  
|      *lite::cv::face::detect::SCRFD*      |     scrfd_10g_bnkps_shape1280x1280.onnx     |     [SCRFD](https://github.com/deepinsight/insightface/blob/master/detection/scrfd)     | 16.9Mb |  
|    *lite::cv::face::detect::YOLO5Face*    |          yolov5face-l-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 181Mb  |
|    *lite::cv::face::detect::YOLO5Face*    |          yolov5face-m-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  |  83Mb  |
|    *lite::cv::face::detect::YOLO5Face*    |        yolov5face-n-0.5-320x320.onnx        |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 2.5Mb  |
|    *lite::cv::face::detect::YOLO5Face*    |        yolov5face-n-0.5-640x640.onnx        |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 4.6Mb  |
|    *lite::cv::face::detect::YOLO5Face*    |          yolov5face-n-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 9.5Mb  |
|    *lite::cv::face::detect::YOLO5Face*    |          yolov5face-s-640x640.onnx          |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  |  30Mb  |
|   *lite::cv::face::detect::FaceBoxesV2*   |          faceboxesv2-640x640.onnx           |                [FaceBoxesV2](https://github.com/jhb86253817/FaceBoxesV2)                | 4.0Mb  |
| *lite::cv::face::detect::YOLOv5BlazeFace* |      yolov5face-blazeface-640x640.onnx      |                 [YOLO5Face](https://github.com/deepcam-cn/yolov5-face)                  | 3.4Mb  |


## Face Alignment.  

<div id="lite.ai.toolkit.hub.onnx-face-alignment"></div>  


|                   Class                    |                     Pretrained ONNX Files                     |                  Rename or Converted From (Repo)                   |  Size   |
|:------------------------------------------:|:-------------------------------------------------------------:|:------------------------------------------------------------------:|:-------:|
|       *lite::cv::face::align::PFLD*        |                      pfld-106-lite.onnx                       | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  1.0Mb  |
|       *lite::cv::face::align::PFLD*        |                       pfld-106-v3.onnx                        | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  5.5Mb  |
|       *lite::cv::face::align::PFLD*        |                       pfld-106-v2.onnx                        | [pfld_106_...](https://github.com/Hsintao/pfld_106_face_landmarks) |  5.0Mb  |
|      *lite::cv::face::align::PFLD98*       |                    PFLD-pytorch-pfld.onnx                     |       [PFLD...](https://github.com/polarisZhao/PFLD-pytorch)       |  4.8Mb  |
|   *lite::cv::face::align::MobileNetV268*   |       pytorch_face_landmarks_landmark_detection_56.onnx       |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  9.4Mb  |
|  *lite::cv::face::align::MobileNetV2SE68*  | pytorch_face_landmarks_landmark_detection_56_se_external.onnx |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  11Mb   |
|      *lite::cv::face::align::PFLD68*       |               pytorch_face_landmarks_pfld.onnx                |  [...landmark](https://github.com/cunjian/pytorch_face_landmark)   |  2.8Mb  |
| *lite::cv::face::align::FaceLandmarks1000* |                     FaceLandmark1000.onnx                     |   [FaceLandm...](https://github.com/Single430/FaceLandmark1000)    |  2.0Mb  |
|     *lite::cv::face::align::PIPNet98*      |            pipnet_resnet18_10x98x32x256_wflw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet68*      |            pipnet_resnet18_10x68x32x256_300w.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet29*      |            pipnet_resnet18_10x29x32x256_cofw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet19*      |            pipnet_resnet18_10x19x32x256_aflw.onnx             |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 44.0Mb  |
|     *lite::cv::face::align::PIPNet98*      |            pipnet_resnet101_10x98x32x256_wflw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet68*      |            pipnet_resnet101_10x68x32x256_300w.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet29*      |            pipnet_resnet101_10x29x32x256_cofw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |
|     *lite::cv::face::align::PIPNet19*      |            pipnet_resnet101_10x19x32x256_aflw.onnx            |          [PIPNet](https://github.com/jhb86253817/PIPNet)           | 150.0Mb |

## Face Attributes.  

<div id="lite.ai.toolkit.hub.onnx-face-attributes"></div>  


|                   Class                   |                    Pretrained ONNX Files                     |                      Rename or Converted From (Repo)                      | Size  |
|:-----------------------------------------:|:------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-----:|
|   *lite::cv::face::attr::AgeGoogleNet*    |                      age_googlenet.onnx                      |               [onnx-models](https://github.com/onnx/models)               | 23Mb  |
|  *lite::cv::face::attr::GenderGoogleNet*  |                    gender_googlenet.onnx                     |               [onnx-models](https://github.com/onnx/models)               | 23Mb  |
|  *lite::cv::face::attr::EmotionFerPlus*   |                    emotion-ferplus-7.onnx                    |               [onnx-models](https://github.com/onnx/models)               | 33Mb  |
|  *lite::cv::face::attr::EmotionFerPlus*   |                    emotion-ferplus-8.onnx                    |               [onnx-models](https://github.com/onnx/models)               | 33Mb  |
|     *lite::cv::face::attr::VGG16Age*      |               vgg_ilsvrc_16_age_imdb_wiki.onnx               |               [onnx-models](https://github.com/onnx/models)               | 514Mb |
|     *lite::cv::face::attr::VGG16Age*      |           vgg_ilsvrc_16_age_chalearn_iccv2015.onnx           |               [onnx-models](https://github.com/onnx/models)               | 514Mb |
|    *lite::cv::face::attr::VGG16Gender*    |             vgg_ilsvrc_16_gender_imdb_wiki.onnx              |               [onnx-models](https://github.com/onnx/models)               | 512Mb |
|      *lite::cv::face::attr::SSRNet*       |                         ssrnet.onnx                          |         [SSR_Net...](https://github.com/oukohou/SSR_Net_Pytorch)          | 190Kb |
| *lite::cv::face::attr::EfficientEmotion7* |           face-emotion-recognition-enet_b0_7.onnx            | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
| *lite::cv::face::attr::EfficientEmotion8* |      face-emotion-recognition-enet_b0_8_best_afew.onnx       | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
| *lite::cv::face::attr::EfficientEmotion8* |      face-emotion-recognition-enet_b0_8_best_vgaf.onnx       | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 15Mb  |
|  *lite::cv::face::attr::MobileEmotion7*   |          face-emotion-recognition-mobilenet_7.onnx           | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 13Mb  |
|  *lite::cv::face::attr::ReXNetEmotion7*   | face-emotion-recognition-affectnet_7_vggface2_rexnet150.onnx | [face-emo...](https://github.com/HSE-asavchenko/face-emotion-recognition) | 30Mb  |


## Face Recognition.  

<div id="lite.ai.toolkit.hub.onnx-face-recognition"></div>  


|                   Class                   |                  Pretrained ONNX Files                  |                    Rename or Converted From (Repo)                     | Size  |
|:-----------------------------------------:|:-------------------------------------------------------:|:----------------------------------------------------------------------:|:-----:|
|     *lite::cv::faceid::GlintArcFace*      |                ms1mv3_arcface_r100.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r50.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 166Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r34.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 130Mb |
|     *lite::cv::faceid::GlintArcFace*      |                 ms1mv3_arcface_r18.onnx                 |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r100.onnx               |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r50.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 166Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r34.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 130Mb |
|     *lite::cv::faceid::GlintCosFace*      |               glint360k_cosface_r18.onnx                |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|    *lite::cv::faceid::GlintPartialFC*     |             partial_fc_glint360k_r100.onnx              |       [insightface](https://github.com/deepinsight/insightface)        | 248Mb |
|    *lite::cv::faceid::GlintPartialFC*     |              partial_fc_glint360k_r50.onnx              |       [insightface](https://github.com/deepinsight/insightface)        | 91Mb  |
|        *lite::cv::faceid::FaceNet*        |              facenet_vggface2_resnet.onnx               |       [facenet...](https://github.com/timesler/facenet-pytorch)        | 89Mb  |
|        *lite::cv::faceid::FaceNet*        |            facenet_casia-webface_resnet.onnx            |       [facenet...](https://github.com/timesler/facenet-pytorch)        | 89Mb  |
|     *lite::cv::faceid::FocalArcFace*      |              focal-arcface-ms1m-ir152.onnx              |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 269Mb |
|     *lite::cv::faceid::FocalArcFace*      |          focal-arcface-ms1m-ir50-epoch120.onnx          |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
|     *lite::cv::faceid::FocalArcFace*      |          focal-arcface-ms1m-ir50-epoch63.onnx           |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
|   *lite::cv::faceid::FocalAsiaArcFace*    |             focal-arcface-bh-ir50-asia.onnx             |   [face.evoLVe...](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)   | 166Mb |
| *lite::cv::faceid::TencentCurricularFace* |          Tencent_CurricularFace_Backbone.onnx           |               [TFace](https://github.com/Tencent/TFace)                | 249Mb |
|    *lite::cv::faceid::TencentCifpFace*    |        Tencent_Cifp_BUPT_Balancedface_IR_34.onnx        |               [TFace](https://github.com/Tencent/TFace)                | 130Mb |
|    *lite::cv::faceid::CenterLossFace*     |              CenterLossFace_epoch_100.onnx              |   [center-loss...](https://github.com/louis-she/center-loss.pytorch)   | 280Mb |
|      *lite::cv::faceid::SphereFace*       |                 sphere20a_20171020.onnx                 |      [sphere...](https://github.com/clcarwin/sphereface_pytorch)       | 86Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |              dream_cfp_res50_end2end.onnx               |             [DREAM](https://github.com/penincillin/DREAM)              | 92Mb  |
|    *lite::cv::faceid::PoseRobustFace*     |              dream_ijba_res18_end2end.onnx              |             [DREAM](https://github.com/penincillin/DREAM)              | 43Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |               dream_cfp_res50_naive.onnx                |             [DREAM](https://github.com/penincillin/DREAM)              | 91Mb  |
|  *lite::cv::faceid:NaivePoseRobustFace*   |               dream_ijba_res18_naive.onnx               |             [DREAM](https://github.com/penincillin/DREAM)              | 43Mb  |
|     *lite::cv::faceid:MobileFaceNet*      |             MobileFaceNet_Pytorch_068.onnx              |   [MobileFace...](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)   | 3.8Mb |
|    *lite::cv::faceid:CavaGhostArcFace*    |      cavaface_GhostNet_x1.3_Arcface_Epoch_24.onnx       |     [cavaface...](https://github.com/cavalleria/cavaface.pytorch)      | 15Mb  |
|    *lite::cv::faceid:CavaCombinedFace*    |        cavaface_IR_SE_100_Combined_Epoch_24.onnx        |     [cavaface...](https://github.com/cavalleria/cavaface.pytorch)      | 250Mb |
|   *lite::cv::faceid:MobileSEFocalFace*    | face_recognition.pytorch_Mobilenet_se_focal_121000.onnx | [face_recog...](https://github.com/grib0ed0v/face_recognition.pytorch) | 4.5Mb |


## Head Pose Estimation.  

<div id="lite.ai.toolkit.hub.onnx-head-pose-estimation"></div>  


|             Class              | Pretrained ONNX Files |                  Rename or Converted From (Repo)                   | Size  |
|:------------------------------:|:---------------------:|:------------------------------------------------------------------:|:-----:|
| *lite::cv::face::pose::FSANet* |    fsanet-var.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |
| *lite::cv::face::pose::FSANet* |    fsanet-1x1.onnx    | [...fsanet...](https://github.com/omasaht/headpose-fsanet-pytorch) | 1.2Mb |


## Segmentation.  

<div id="lite.ai.toolkit.hub.onnx-segmentation"></div>  


|                       Class                       |          Pretrained ONNX Files          |                         Rename or Converted From (Repo)                          | Size  |
|:-------------------------------------------------:|:---------------------------------------:|:--------------------------------------------------------------------------------:|:-----:|
|   *lite::cv::segmentation::DeepLabV3ResNet101*    |      deeplabv3_resnet101_coco.onnx      |                 [torchvision](https://github.com/pytorch/vision)                 | 232Mb |
|      *lite::cv::segmentation::FCNResNet101*       |           fcn_resnet101.onnx            |                 [torchvision](https://github.com/pytorch/vision)                 | 207Mb |
|         *lite::cv::segmentation::HeadSeg*         |        minivision_head_seg.onnx         |         [photo2cartoon](https://github.com/minivision-ai/photo2cartoon)          | 31Mb  |
|     *lite::cv::segmentation::FastPortraitSeg*     | fast_portrait_seg_SINet_bi_192_128.onnx |   [Fast-Portrait...](https://github.com/YexingWan/Fast-Portrait-Segmentation)    | 400k  |
|     *lite::cv::segmentation::FastPortraitSeg*     | fast_portrait_seg_SINet_bi_256_160.onnx |   [Fast-Portrait...](https://github.com/YexingWan/Fast-Portrait-Segmentation)    | 400k  |
|     *lite::cv::segmentation::FastPortraitSeg*     | fast_portrait_seg_SINet_bi_320_256.onnx |   [Fast-Portrait...](https://github.com/YexingWan/Fast-Portrait-Segmentation)    | 400k  |
|    *lite::cv::segmentation::PortraitSegSINet*     |   ext_portrait_seg_SINet_224x224.onnx   |     [ext_portrait...](https://github.com/clovaai/ext_portrait_segmentation)      | 380k  |
| *lite::cv::segmentation::PortraitSegExtremeC3Net* | ext_portrait_seg_ExtremeC3_224x224.onnx |     [ext_portrait...](https://github.com/clovaai/ext_portrait_segmentation)      | 180k  |
|       *lite::cv::segmentation::FaceHairSeg*       |       face_hair_seg_224x224.onnx        |                  [face-seg](https://github.com/kampta/face-seg)                  |  18M  |
|         *lite::cv::segmentation::HairSeg*         |          hairseg_224x224.onnx           | [mobile-semantic-seg](https://github.com/akirasosa/mobile-semantic-segmentation) |  18M  |


## Style Transfer.  

<div id="lite.ai.toolkit.hub.onnx-style-transfer"></div>  

|                 Class                  |        Pretrained ONNX Files         |                 Rename or Converted From (Repo)                 | Size  |
|:--------------------------------------:|:------------------------------------:|:---------------------------------------------------------------:|:-----:|
|  *lite::cv::style::FastStyleTransfer*  |         style-mosaic-8.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |          style-candy-9.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |          style-udnie-8.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |          style-udnie-9.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |       style-pointilism-8.onnx        |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |       style-pointilism-9.onnx        |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |      style-rain-princess-9.onnx      |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |      style-rain-princess-8.onnx      |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |          style-candy-8.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
|  *lite::cv::style::FastStyleTransfer*  |         style-mosaic-9.onnx          |          [onnx-models](https://github.com/onnx/models)          | 6.4Mb |
| *lite::cv::style::FemalePhoto2Cartoon* | minivision_female_photo2cartoon.onnx | [photo2cartoon](https://github.com/minivision-ai/photo2cartoon) | 15Mb  |


## Colorization.  

<div id="lite.ai.toolkit.hub.onnx-colorization"></div>

|                Class                |   Pretrained ONNX Files   |              Rename or Converted From (Repo)              | Size  |
|:-----------------------------------:|:-------------------------:|:---------------------------------------------------------:|:-----:|
| *lite::cv::colorization::Colorizer* |   eccv16-colorizer.onnx   | [colorization](https://github.com/richzhang/colorization) | 123Mb |
| *lite::cv::colorization::Colorizer* | siggraph17-colorizer.onnx | [colorization](https://github.com/richzhang/colorization) | 129Mb |


## Super Resolution.  

<div id="lite.ai.toolkit.hub.onnx-super-resolution"></div>

|                Class                | Pretrained ONNX Files |              Rename or Converted From (Repo)              | Size  |
|:-----------------------------------:|:---------------------:|:---------------------------------------------------------:|:-----:|
| *lite::cv::resolution::SubPixelCNN* |   subpixel-cnn.onnx   | [...PIXEL...](https://github.com/niazwazir/SUB_PIXEL_CNN) | 234Kb |


## Matting.

<div id="lite.ai.toolkit.hub.onnx-matting"></div>

|                    Class                    |                Pretrained ONNX Files                |                    Rename or Converted From (Repo)                     | Size  |
|:-------------------------------------------:|:---------------------------------------------------:|:----------------------------------------------------------------------:|:-----:|
|   *lite::cv::matting::RobustVideoMatting*   |              rvm_mobilenetv3_fp32.onnx              |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 14Mb  |
|   *lite::cv::matting::RobustVideoMatting*   |              rvm_mobilenetv3_fp16.onnx              |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 7.2Mb |
|   *lite::cv::matting::RobustVideoMatting*   |               rvm_resnet50_fp32.onnx                |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 50Mb  |
|   *lite::cv::matting::RobustVideoMatting*   |               rvm_resnet50_fp16.onnx                |  [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)  | 100Mb |
|       *lite::cv::matting::MGMatting*        |               MGMatting-DIM-100k.onnx               |          [MGMatting](https://github.com/yucornetto/MGMatting)          | 113Mb |
|       *lite::cv::matting::MGMatting*        |               MGMatting-RWP-100k.onnx               |          [MGMatting](https://github.com/yucornetto/MGMatting)          | 113Mb |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-1024x1024.onnx |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-1024x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-256x256.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-256x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         | modnet_photographic_portrait_matting-512x1024.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-512x256.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |  modnet_photographic_portrait_matting-512x512.onnx  |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-1024x1024.onnx    |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-1024x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-256x256.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-256x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |    modnet_webcam_portrait_matting-512x1024.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-512x256.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|         *lite::cv::matting::MODNet*         |     modnet_webcam_portrait_matting-512x512.onnx     |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|       *lite::cv::matting::MODNetDyn*        |      modnet_photographic_portrait_matting.onnx      |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
|       *lite::cv::matting::MODNetDyn*        |         modnet_webcam_portrait_matting.onnx         |               [MODNet](https://github.com/ZHKKKe/MODNet)               | 24Mb  |
  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_mobilenetv2-256x256-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_mobilenetv2-512x512-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |        BGMv2_mobilenetv2-1080x1920-full.onnx        | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |        BGMv2_mobilenetv2-2160x3840-full.onnx        | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet50-1080x1920-full.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet50-2160x3840-full.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 20Mb  |
|  *lite::cv::matting::BackgroundMattingV2*   |         BGMv2_resnet101-2160x3840-full.onnx         | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 154Mb |
| *lite::cv::matting::BackgroundMattingV2Dyn* |          BGMv2_mobilenetv2_4k_dynamic.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 157Mb |
| *lite::cv::matting::BackgroundMattingV2Dyn* |          BGMv2_mobilenetv2_hd_dynamic.onnx          | [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 230Mb |

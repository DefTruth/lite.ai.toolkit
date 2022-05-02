# Linux 入門教程

* [編譯 Lite.Ai.ToolKit](#lite.ai.toolkit-linuxzh-Self-Compile)
* [在自己的專案中使用 Lite.Ai.ToolKit](#lite.ai.toolkit-linuxzh-Sample-Project)

## 編譯 Lite.Ai.ToolKit
<div id="lite.ai.toolkit-linuxzh-Self-Compile"></div>

在 Linux 之下，編譯 lite.ai.toolkit 共有四個步驟
1. 編譯 FFmpeg
2. 編譯 OpenCV
3. 下載 Onnx Runtime 動態庫
4. 編譯 Lite.Ai.ToolKit

### 1. FFmpeg
```
mkdir ~/library && cd ~/library
git clone --depth=1 https://git.ffmpeg.org/ffmpeg.git -b n4.2.2
cd ffmpeg
./configure --enable-shared --disable-x86asm --prefix=/usr/local/opt/ffmpeg --disable-static
make -j8
sudo make install
```

### 2. OpenCV
下載原始碼
```
cd ~/library
wget https://github.com/opencv/opencv/archive/refs/tags/4.5.2.zip
unzip 4.5.2.zip
```
編譯安裝
```
cd opencv-4.5.2
mkdir build && cd build
cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=your-path-to-custom-dir \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_python3=OFF \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_opencv_apps=OFF \
  -D WITH_FFMPEG=ON 
make -j8
sudo make install
```
查詢已安裝的 opencv 版本
```
opencv_version
> 4.5.2
```
**經過測試目前不支援最新版 v4.5.5**
### 3. ONNX Runtime
下載官方建構的動態庫(不需要再從原始碼進行編譯)
```
cd ~/library
wget https://github.com/microsoft/onnxruntime/releases/download/v1.7.0/onnxruntime-linux-x64-1.7.0.tgz
tar zxvf onnxruntime-linux-x64-1.7.0.tgz
```
可以看到 lib 下面有編譯好的動態庫
```
ls onnxruntime-linux-x64-1.7.0/lib
> libonnxruntime.so  libonnxruntime.so.1.7.0
```

### 4. Lite.Ai.ToolKit
<div id="lite.ai.toolkit-linuxzh-Compile-Lite.Ai.Toolkit"></div>

下載原始碼
```
cd ~/library
git clone https://github.com/DefTruth/lite.ai.toolkit.git
```
* 複製自行編譯的 opencv 動態庫到 lib/linux 之下
* 複製官方編譯的 onnxruntime 動態庫到 lib/linux 之下
```
cd lite.ai.toolkit
cp ~/library/opencv-4.5.2/build/lib/* lib/linux
cp ~/library/onnxruntime-linux-x64-1.7.0/lib/* lib/linux
```
編譯 lite.ai.toolkit
```
sh ./build.sh
```
執行測試範例
```
cd build/lite.ai.toolkit/bin
./lite_yolov5
```
以下為輸出 LOG，如果執行成功，偵測結果會被存在 logs 之下
```
LITEORT_DEBUG LogId: ../../../hub/onnx/cv/yolov5s.onnx
=============== Input-Dims ==============
input_node_dims: 1
input_node_dims: 3
input_node_dims: 640
input_node_dims: 640
=============== Output-Dims ==============
Output: 0 Name: pred Dim: 0 :1
Output: 0 Name: pred Dim: 1 :25200
Output: 0 Name: pred Dim: 2 :85
Output: 1 Name: output2 Dim: 0 :1
Output: 1 Name: output2 Dim: 1 :3
Output: 1 Name: output2 Dim: 2 :80
Output: 1 Name: output2 Dim: 3 :80
Output: 1 Name: output2 Dim: 4 :85
Output: 2 Name: output3 Dim: 0 :1
Output: 2 Name: output3 Dim: 1 :3
Output: 2 Name: output3 Dim: 2 :40
Output: 2 Name: output3 Dim: 3 :40
Output: 2 Name: output3 Dim: 4 :85
Output: 3 Name: output4 Dim: 0 :1
Output: 3 Name: output4 Dim: 1 :3
Output: 3 Name: output4 Dim: 2 :20
Output: 3 Name: output4 Dim: 3 :20
Output: 3 Name: output4 Dim: 4 :85
========================================
detected num_anchors: 25200
generate_bboxes num: 39
ONNXRuntime Version Detected Boxes Num: 4
```
## 在自己的專案中使用 Lite.Ai.ToolKit
<div id="lite.ai.toolkit-linuxzh-Sample-Project"></div>

這邊我們將從零開始，建立一個屬於自己的應用程式專案

最關鍵的就是如何使用 Lite.Ai.ToolKit

整個專案共分成四個步驟
1. 連結函式庫&動態庫
2. 編寫程式碼
3. 編譯專案
4. 執行應用程式

_補充：本教程都是以 ONNX Runtime 為主，MNN, ncnn, tnn 請自行嘗試_

### 1. 連結函式庫&動態庫
首先，先建立專案資料夾，本教程使用 SCRFD(人脸检测) 當範例
```
mkdir -p ~/ORT_Projects/SCRFD
cd ~/ORT_Projects/SCRFD
```
複製[之前編譯教程](#lite.ai.toolkit-linuxzh-Compile-Lite.Ai.Toolkit)編譯好的 include 跟 lib 到 SCRFD/lite.ai.toolkit
```
mkdir lite.ai.toolkit
cp -r ~/library/lite.ai.toolkit/build/lite.ai.toolkit/include lite.ai.toolkit
cp -r ~/library/lite.ai.toolkit/build/lite.ai.toolkit/lib lite.ai.toolkit
```
設置環境變數，可以直接執行這兩行，也可以寫入 `.bashrc` 或 `.zshrc`
```
export LD_LIBRARY_PATH=~/ORT-Projects/SCRFD/lite.ai.toolkit/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=~/ORT-Projects/SCRFD/lite.ai.toolkit/lib:$LIBRARY_PATH
```

### 2. 編寫程式碼
建立主程式 `scrfd.cpp`，程式內容如下
```C++
#include "lite/lite.h"

static void test_lite()
{
    std::string onnx_path = "../scrfd_2.5g_bnkps_shape640x640.onnx";
    std::string test_img_path = "../test_lite_face_detector_2.jpg";
    std::string save_img_path = "../test_lite_scrfd.jpg";
    
    auto *scrfd = new lite::cv::face::detect::SCRFD(onnx_path);
		  
    std::vector<lite::types::BoxfWithLandmarks> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    scrfd->detect(img_bgr, detected_boxes);
		        
    lite::utils::draw_boxes_with_landmarks_inplace(img_bgr, detected_boxes);
    cv::imwrite(save_img_path, img_bgr);
			    
    std::cout << "Default Version Done! Detected Face Num: " << detected_boxes.size() << std::endl;
			      
    delete scrfd;
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}
```
程式所需要的輸入檔請至以下連結下載
* onnx_path: [scrfd_2.5g_bnkps_shape640x640.onnx](https://github.com/DefTruth/lite.ai.toolkit#5-model-zoo)
* test_img_path: [test_lite_face_detector_2.jpg](https://github.com/DefTruth/lite.ai.toolkit/blob/main/examples/lite/resources/test_lite_face_detector_2.jpg)

### 3. 編譯專案 
使用 CMake 編譯專案會比較方便，`CMakeLists.txt` 內容如下
```cmake
cmake_minimum_required(VERSION 3.17)
project(scrfd)

set(CMAKE_CXX_STANDARD 11)

set(LITE_AI_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
include_directories(${LITE_AI_DIR}/include)
link_directories(${LITE_AI_DIR}/lib})

set(TOOLKIT_LIBS lite.ai.toolkit onnxruntime)
set(OpenCV_LIBS opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui)

add_executable(test_scrfd scrfd.cpp)
target_link_libraries(test_scrfd ${TOOLKIT_LIBS} ${OpenCV_LIBS})
```
編譯專案
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Realese .. && make -j1
```

### 4. 執行應用程式
第三步驟如果成功了，在 build 下面應該會有一個 `test_scrfd`，執行它進行測試
```
./test_scrfd
```
輸出 `../test_lite_scrfd.jpg` 結果圖，可以看到成功檢測出人臉

![test_lite_scrfd](https://user-images.githubusercontent.com/91650059/165025024-00cc2c26-8478-454f-be7a-0e23a164054f.jpg)

## 说明
以上案例是基于最小依赖，即opencv和onnxruntime实现的，如果需要MNN、TNN和NCNN等推理引擎的支持，请参考项目README.md文档中的编译选项进行配置。
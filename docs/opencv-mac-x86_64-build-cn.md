# MacOS源码编译OpenCV
## 1.使用Homebrew安装cmake
```shell
brew install cmake
# 检查cmake和make
which cmake
cmake --version
which make
```
## 2.获取OpenCV源代码
```shell
# 科学上网 使用cnpmjs.org加速拉取 亲测有效
git clone --depth=1 https://github.com.cnpmjs.org/opencv/opencv.git
```

## 3.源码编译
```shell
mkdir build
cd build
cmake ..
```
cmake配置完成之后，发现opencv的默认安装目录为`usr/local`：
```shell
--
--   Python (for build):            /usr/bin/python2.7
--
--   Java:
--     ant:                         NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
--
--   Install to:                    /usr/local
```
但是我们在工程中，通常需要将opencv打包到app，因此不建议将opencv直接安装在系统目录。编译时指定编译`Release`版本，并且指定自定义的安装目录。
```shell
# 自定义的安装路径
CUSTOM_INSTALL_PREFIX=/Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64/

# 更新cmake的设置
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CUSTOM_INSTALL_PREFIX}
```
更新之后opencv的安装路径为：
```shell
--   Python (for build):            /usr/bin/python2.7
--
--   Java:
--     ant:                         NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
--
--   Install to:                    /Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64
-- -----------------------------------------------------------------
```
剩下的就是常规的编译过程了：
```shell
# 编译
make -j16

# 安装
make install
```
安装的log如下：
```shell
...
-- Installing: /Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64/include/opencv4/opencv2/gapi/infer/bindings_ie.hpp
-- Installing: /Users/xxx/xxx//third_party/library/opencv/libs/built-20210416/x86_64/include/opencv4/opencv2/gapi/infer/ie.hpp
-- Installing: /Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64/include/opencv4/opencv2/gapi/infer/onnx.hpp
-- Installing: /Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64/include/opencv4/opencv2/gapi/infer/parsers.hpp
-- Installing: /Users/xxx/xxx/third_party/library/opencv/libs/built-20210416/x86_64/include/opencv4/opencv2/gapi/media.hpp
...
```
可以看到opencv已经安装在指定的目录下了，需要注意的是，你必须执行`make install`，cmake才会将需要用到的头文件拷贝到`include`文件夹中，否则，该文件夹为空。
```shell
cd ${CUSTOM_INSTALL_PREFIX}
ls
# bin     include lib     share
```

## 4.Hello World!
```c++
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::string;

int main(int argc, const char * argv[]) {
    string path = "/Users/*/Desktop/test.jpg";
    Mat image = imread(path);
    namedWindow("origin");
    imshow("origin", image);
    
    Mat gray;
    cvtColor(image, gray, COLOR_RGBA2GRAY);
    namedWindow("gray");
    imshow("gray", gray);
    waitKey(0);

    return 0;
}
```  
文档内容可以在[opencv-mac-x86_64-build-cn.md](https://github.com/DefTruth/litehub/blob/main/docs/opencv-mac-x86_64-build-cn.md) 中找到。
# Linux下TensorRT环境配置

## 1.环境准备

首先确定本机是否存在CUDA和cudnn环境(必须),文档默认用户已经正常安装CUDA和cudnn,
测试本机的CUDA版本 在shell中输入nvcc -V即可，这里建议CUDA为12.4+cudnn9.1.0.70+TensorRT10.1.0.27

## 2.TensorRT安装包下载

在确定本机环境中存在CUDA和cudnn之后,就可以开始配置TensorRT的环境了,首先需要去Nvidia
官网下载TensorRT的安装包 这里需要注册一个Nvidia的账号才能下载,可以用Google账号登陆,
本项目用的TensorRT版本为10.1.0.27
千万不要去下载CSDN上的付费(真的很难绷)资源 后期有时间的话会将本项目的TensorRT的安装包上传到百度网盘或者谷歌网盘,这里要注意的是CUDA和cudnn和TensorRT的版本必须对应不然没办法使用,
其实还是预备环境需要完成配置,如果CUDA和cudnn配置好的话,只需要对应好TensorRT和CUDA的版本就好了,本项目配置的是CUDA12.4+TensorRT10.1.0.27版本测试是OK的

## 3.TensorRT环境配置
下载完成之后,直接将其解压对应的位置,这里推荐解压到/usr/local的目录下和cuda在同一级目录下, 并重命名为tensorrt，当然这个看个人喜好,你想解压在哪里都是可以的,解压完成之后需要将解压位置加入到环境变量当中,
需要编辑你的环境配置文件bashrc
```shell
vim ~/.bashrc
# 将你TensorRT的安装位置加入到bashrc当中
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/tensorrt/bin:$PATH
# 加完之后需要source一下bashrc
source ~/.bashrc
```
安装完成之后测试TensorRT能否正常工作可以去/path/to/tensorrt/samples/sampleOnnxMNIST路径下执行make命令 如果正常执行完成会出现可执行文件的路径一般是../bin,执行命令
```shell
./sample_onnx_mnist
```
如果有输出则说明安装成功  

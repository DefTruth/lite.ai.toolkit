//
// Created by DefTruth on 2021/4/5.
//

#include <iostream>
#include <vector>

#include "ort/cv/fast_style_transfer.h"
#include "ort/core/ort_utils.h"

static void test_ortcv_fast_style_transfer()
{
  std::string candy_onnx_path = "../../../hub/onnx/cv/style-candy-8.onnx";
  std::string mosaic_onnx_path = "../../../hub/onnx/cv/style-mosaic-8.onnx";
  std::string pointilism_onnx_path = "../../../hub/onnx/cv/style-pointilism-8.onnx";
  std::string rain_princess_onnx_path = "../../../hub/onnx/cv/style-rain-princess-8.onnx";
  std::string udnie_onnx_path = "../../../hub/onnx/cv/style-udnie-8.onnx";
  std::string test_img_path = "../../../examples/ort/resources/test_ortcv_fast_style_transfer.jpg";
  std::string save_candy_path = "../../../logs/test_ortcv_fast_style_transfer_candy.jpg";
  std::string save_mosaic_path = "../../../logs/test_ortcv_fast_style_transfer_mosaic.jpg";
  std::string save_pointilism_path = "../../../logs/test_ortcv_fast_style_transfer_pointilism.jpg";
  std::string save_rain_princess_path = "../../../logs/test_ortcv_fast_style_transfer_rain_princes.jpg";
  std::string save_udnie_path = "../../../logs/test_ortcv_fast_style_transfer_udnie.jpg";

  ortcv::FastStyleTransfer *candy_fast_style_transfer = new ortcv::FastStyleTransfer(candy_onnx_path);
  ortcv::FastStyleTransfer *mosaic_fast_style_transfer = new ortcv::FastStyleTransfer(mosaic_onnx_path);
  ortcv::FastStyleTransfer *pointilism_fast_style_transfer = new ortcv::FastStyleTransfer(pointilism_onnx_path);
  ortcv::FastStyleTransfer *rain_princess_fast_style_transfer = new ortcv::FastStyleTransfer(rain_princess_onnx_path);
  ortcv::FastStyleTransfer *udnie_fast_style_transfer = new ortcv::FastStyleTransfer(udnie_onnx_path);

  ortcv::types::StyleContent candy_style_content;
  ortcv::types::StyleContent mosaic_style_content;
  ortcv::types::StyleContent pointilism_style_content;
  ortcv::types::StyleContent rain_princess_style_content;
  ortcv::types::StyleContent udnie_style_content;

  cv::Mat img_bgr = cv::imread(test_img_path);

  candy_fast_style_transfer->detect(img_bgr, candy_style_content);
  mosaic_fast_style_transfer->detect(img_bgr, mosaic_style_content);
  pointilism_fast_style_transfer->detect(img_bgr, pointilism_style_content);
  rain_princess_fast_style_transfer->detect(img_bgr, rain_princess_style_content);
  udnie_fast_style_transfer->detect(img_bgr, udnie_style_content);

  if (candy_style_content.flag) cv::imwrite(save_candy_path, candy_style_content.mat);
  if (mosaic_style_content.flag) cv::imwrite(save_mosaic_path, mosaic_style_content.mat);
  if (pointilism_style_content.flag) cv::imwrite(save_pointilism_path, pointilism_style_content.mat);
  if (rain_princess_style_content.flag) cv::imwrite(save_rain_princess_path, rain_princess_style_content.mat);
  if (udnie_style_content.flag) cv::imwrite(save_udnie_path, udnie_style_content.mat);

  std::cout << "Style Transfer Done." << std::endl;

  delete candy_fast_style_transfer;
  delete mosaic_fast_style_transfer;
  delete pointilism_fast_style_transfer;
  delete rain_princess_fast_style_transfer;
  delete udnie_fast_style_transfer;
}

int main(__unused int argc, __unused char *argv[])
{
  test_ortcv_fast_style_transfer();
  return 0;
}
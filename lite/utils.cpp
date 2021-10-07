//
// Created by DefTruth on 2021/10/7.
//

#include "utils.h"

//*************************************** lite::utils **********************************************//
std::string lite::utils::to_string(const std::wstring &wstr)
{
  unsigned len = wstr.size() * 4;
  setlocale(LC_CTYPE, "");
  char *p = new char[len];
  wcstombs(p, wstr.c_str(), len);
  std::string str(p);
  delete[] p;
  return str;
}

std::wstring lite::utils::to_wstring(const std::string &str)
{
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t *p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

// reference: https://github.com/DefTruth/headpose-fsanet-pytorch/blob/master/src/utils.py
void lite::utils::draw_axis_inplace(cv::Mat &mat_inplace,
                                     const types::EulerAngles &euler_angles,
                                     float size, int thickness)
{
  if (!euler_angles.flag) return;

  const float pitch = euler_angles.pitch * _PI / 180.f;
  const float yaw = -euler_angles.yaw * _PI / 180.f;
  const float roll = euler_angles.roll * _PI / 180.f;

  const float height = static_cast<float>(mat_inplace.rows);
  const float width = static_cast<float>(mat_inplace.cols);

  const int tdx = static_cast<int>(width / 2.0f);
  const int tdy = static_cast<int>(height / 2.0f);

  // X-Axis pointing to right. drawn in red
  const int x1 = static_cast<int>(size * std::cosf(yaw) * std::cosf(roll)) + tdx;
  const int y1 = static_cast<int>(
                     size * (std::cosf(pitch) * std::sinf(roll)
                             + std::cosf(roll) * std::sinf(pitch) * std::sinf(yaw))
                 ) + tdy;
  // Y-Axis | drawn in green
  const int x2 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(roll)) + tdx;
  const int y2 = static_cast<int>(
                     size * (std::cosf(pitch) * std::cosf(roll)
                             - std::sinf(pitch) * std::sinf(yaw) * std::sinf(roll))
                 ) + tdy;
  // Z-Axis (out of the screen) drawn in blue
  const int x3 = static_cast<int>(size * std::sinf(yaw)) + tdx;
  const int y3 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(pitch)) + tdy;

  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
  cv::line(mat_inplace, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);
}

cv::Mat lite::utils::draw_axis(const cv::Mat &mat,
                                const types::EulerAngles &euler_angles,
                                float size, int thickness)
{
  if (!euler_angles.flag) return mat;

  cv::Mat mat_copy = mat.clone();
  const float pitch = euler_angles.pitch * _PI / 180.f;
  const float yaw = -euler_angles.yaw * _PI / 180.f;
  const float roll = euler_angles.roll * _PI / 180.f;

  const float height = static_cast<float>(mat_copy.rows);
  const float width = static_cast<float>(mat_copy.cols);

  const int tdx = static_cast<int>(width / 2.0f);
  const int tdy = static_cast<int>(height / 2.0f);

  // X-Axis pointing to right. drawn in red
  const int x1 = static_cast<int>(size * std::cosf(yaw) * std::cosf(roll)) + tdx;
  const int y1 = static_cast<int>(
                     size * (std::cosf(pitch) * std::sinf(roll)
                             + std::cosf(roll) * std::sinf(pitch) * std::sinf(yaw))
                 ) + tdy;
  // Y-Axis | drawn in green
  const int x2 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(roll)) + tdx;
  const int y2 = static_cast<int>(
                     size * (std::cosf(pitch) * std::cosf(roll)
                             - std::sinf(pitch) * std::sinf(yaw) * std::sinf(roll))
                 ) + tdy;
  // Z-Axis (out of the screen) drawn in blue
  const int x3 = static_cast<int>(size * std::sinf(yaw)) + tdx;
  const int y3 = static_cast<int>(-size * std::cosf(yaw) * std::sinf(pitch)) + tdy;

  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x1, y1), cv::Scalar(0, 0, 255), thickness);
  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x2, y2), cv::Scalar(0, 255, 0), thickness);
  cv::line(mat_copy, cv::Point2i(tdx, tdy), cv::Point2i(x3, y3), cv::Scalar(255, 0, 0), thickness);

  return mat_copy;
}

cv::Mat lite::utils::draw_landmarks(const cv::Mat &mat, types::Landmarks &landmarks)
{
  if (landmarks.points.empty() || !landmarks.flag) return mat;
  cv::Mat mat_copy = mat.clone();
  for (const auto &point: landmarks.points)
    cv::circle(mat_copy, point, 2, cv::Scalar(0, 255, 0), -1);
  return mat_copy;
}

void lite::utils::draw_landmarks_inplace(cv::Mat &mat, types::Landmarks &landmarks)
{
  if (landmarks.points.empty() || !landmarks.flag) return;
  for (const auto &point: landmarks.points)
    cv::circle(mat, point, 2, cv::Scalar(0, 255, 0), -1);
}

void lite::utils::draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes)
{
  if (boxes.empty()) return;
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6f, cv::Scalar(0, 255, 0), 2);

      }
    }
  }
}

cv::Mat lite::utils::draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes)
{
  if (boxes.empty()) return mat;
  cv::Mat canva = mat.clone();
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(canva, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(canva, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6f, cv::Scalar(0, 255, 0), 2);

      }
    }
  }
  return canva;
}

cv::Mat lite::utils::draw_age(const cv::Mat &mat, types::Age &age)
{
  if (!age.flag) return mat;
  cv::Mat canva = mat.clone();
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat.rows));
  std::string age_text = "Age:" + std::to_string(age.age).substr(0, 4);
  std::string interval_text = "Interval" + std::to_string(age.age_interval[0])
                              + "_" + std::to_string(age.age_interval[1]);
  std::string interval_prob = "Prob:" + std::to_string(age.interval_prob).substr(0, 4);
  cv::putText(canva, age_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
  cv::putText(canva, interval_text, cv::Point2i(10, offset * 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(255, 0, 0), 2);
  cv::putText(canva, interval_prob, cv::Point2i(10, offset * 3),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 0, 255), 2);
  return canva;
}

void lite::utils::draw_age_inplace(cv::Mat &mat_inplace, types::Age &age)
{
  if (!age.flag) return;
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat_inplace.rows));
  std::string age_text = "Age:" + std::to_string(age.age).substr(0, 4);
  std::string interval_text = "Interval" + std::to_string(age.age_interval[0])
                              + "_" + std::to_string(age.age_interval[1]);
  std::string interval_prob = "Prob:" + std::to_string(age.interval_prob).substr(0, 4);
  cv::putText(mat_inplace, age_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
  cv::putText(mat_inplace, interval_text, cv::Point2i(10, offset * 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(255, 0, 0), 2);
  cv::putText(mat_inplace, interval_prob, cv::Point2i(10, offset * 3),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 0, 255), 2);
}

cv::Mat lite::utils::draw_gender(const cv::Mat &mat, types::Gender &gender)
{
  if (!gender.flag) return mat;
  cv::Mat canva = mat.clone();
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat.rows));
  std::string gender_text = std::to_string(gender.label) + ":"
                            + gender.text + ":" + std::to_string(gender.score).substr(0, 4);
  cv::putText(canva, gender_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
  return canva;
}

void lite::utils::draw_gender_inplace(cv::Mat &mat_inplace, types::Gender &gender)
{
  if (!gender.flag) return;
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat_inplace.rows));
  std::string gender_text = std::to_string(gender.label) + ":"
                            + gender.text + ":" + std::to_string(gender.score).substr(0, 4);
  cv::putText(mat_inplace, gender_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
}

cv::Mat lite::utils::draw_emotion(const cv::Mat &mat, types::Emotions &emotions)
{
  if (!emotions.flag) return mat;
  cv::Mat canva = mat.clone();
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat.rows));
  std::string emotion_text = std::to_string(emotions.label) + ":"
                             + emotions.text + ":" + std::to_string(emotions.score).substr(0, 4);
  cv::putText(canva, emotion_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
  return canva;
}

void lite::utils::draw_emotion_inplace(cv::Mat &mat_inplace, types::Emotions &emotions)
{
  if (!emotions.flag) return;
  const unsigned int offset = static_cast<unsigned int>(
      0.1f * static_cast<float>(mat_inplace.rows));
  std::string emotion_text = std::to_string(emotions.label) + ":"
                             + emotions.text + ":" + std::to_string(emotions.score).substr(0, 4);
  cv::putText(mat_inplace, emotion_text, cv::Point2i(10, offset),
              cv::FONT_HERSHEY_SIMPLEX, 0.6f, cv::Scalar(0, 255, 0), 2);
}

// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
//            blob/master/ncnn/src/UltraFace.cpp
void lite::utils::hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                            float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

void lite::utils::blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                                float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }

    float total = 0.f;
    for (unsigned int k = 0; k < buf.size(); ++k)
    {
      total += std::expf(buf[k].score);
    }
    types::Boxf rects;
    for (unsigned int l = 0; l < buf.size(); ++l)
    {
      float rate = std::expf(buf[l].score) / total;
      rects.x1 += buf[l].x1 * rate;
      rects.y1 += buf[l].y1 * rate;
      rects.x2 += buf[l].x2 * rate;
      rects.y2 += buf[l].y2 * rate;
      rects.score += buf[l].score * rate;
    }
    rects.flag = true;
    output.push_back(rects);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void lite::utils::offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                              float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  const float offset = 4096.f;
  /** Add offset according to classes.
   * That is, separate the boxes into categories, and each category performs its
   * own NMS operation. The same offset will be used for those predicted to be of
   * the same category. Therefore, the relative positions of boxes of the same
   * category will remain unchanged. Box of different classes will be farther away
   * after offset, because offsets are different. In this way, some overlapping but
   * different categories of entities are not filtered out by the NMS. Very clever!
   */
  for (unsigned int i = 0; i < box_num; ++i)
  {
    input[i].x1 += static_cast<float>(input[i].label) * offset;
    input[i].y1 += static_cast<float>(input[i].label) * offset;
    input[i].x2 += static_cast<float>(input[i].label) * offset;
    input[i].y2 += static_cast<float>(input[i].label) * offset;
  }

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }

  /** Substract offset.*/
  if (!output.empty())
  {
    for (unsigned int i = 0; i < output.size(); ++i)
    {
      output[i].x1 -= static_cast<float>(output[i].label) * offset;
      output[i].y1 -= static_cast<float>(output[i].label) * offset;
      output[i].x2 -= static_cast<float>(output[i].label) * offset;
      output[i].y2 -= static_cast<float>(output[i].label) * offset;
    }
  }

}


//*************************************** lite::utils **********************************************//

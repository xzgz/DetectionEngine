#ifndef C_OBJECT_FLOW_TEST_H
#define C_OBJECT_FLOW_TEST_H

#include <iostream>
#include <utility>
#include <cmath>
#include <iomanip>
#include <sys/time.h>

#include "utils/bbox.h"

using std::cout;
using std::endl;
using std::pair;
using std::min;
using std::max;
using std::vector;
using std::string;
using namespace cobjectflow;

#define EPS 1e-6

static double what_time_is_it_now()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

class Box {
 public:
  float x1, y1, x2, y2;
  int label;

  Box(const vector<float> & vec) {
    if (vec.size() > 0) x1 = vec[0];
    if (vec.size() > 1) y1 = vec[1];
    if (vec.size() > 2) x2 = vec[2];
    if (vec.size() > 3) y2 = vec[3];
    label = 0;
  }

  Box(float left = 0, float top = 0, float right = 0, float bottom = 0, int new_label = 0) {
    x1 = left;
    y1 = top;
    x2 = right;
    y2 = bottom;
    label = new_label;
  }

  // operator overload
  Box operator + (const Box & b) { // box add box
    return Box(x1 + b.x1, y1 + b.y1, x2 + b.x2, y2 + b.y2, label);
  }
  Box operator - (const Box & b) { // box minux box
    return Box(x1 - b.x1, y1 - b.y1, x2 - b.x2, y2 - b.y2, label);
  }
  Box operator + (const pair<float, float> & p) { // box add point
    return Box(x1 + p.first, y1 + p.second, x2 + p.first, y2 + p.second, label);
  }
  Box operator - (const pair<float, float> & p) { // box minux point
    return Box(x1 - p.first, y1 - p.second, x2 - p.first, y2 - p.second, label);
  }
  Box operator * (const float scale) {
    return Box(x1 * scale, y1 * scale, x2 * scale, y2 * scale, label);
  }
  Box operator / (const float scale) {
    if (scale > EPS || scale < -EPS)
      return Box(x1 / scale, y1 / scale, x2 / scale, y2 / scale, label);
  }
  bool operator == (const Box & b) {
    return label == b.label
           && x1 == b.x1 && x2 == b.x2
           && y1 == b.y1 && y2 == b.y2;
  }

  // attributes
  float width() const {
    return x2 - x1 + 1;
  }

  float height() const {
    return y2 - y1 + 1;
  }

  float center_x() const {
    return x1 + width() * 0.5;
  }

  float center_y() const {
    return y1 + height() * 0.5;
  }

  float area() const {
    if (x1 > x2 || y1 > y2) return 0;
    return width() * height();
  }

  float diag() const {
    return sqrt(width() * width() + height() * height());
  }

  float getIoU(const Box & b) const {
    float left = std::max(x1, b.x1);
    float top = std::max(y1, b.y1);
    float right = std::min(x2, b.x2);
    float bottom = std::min(y2, b.y2);
    if (left > right || top > bottom) return 0;
    float inter = (right - left + 1) * (bottom - top + 1);
    if ((area() + b.area() - inter) <= 0) return 0;
    return inter / (area() + b.area() - inter);
  }

  void show() const {
    cout << "Showing Box" << endl;
    cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
  } // show()
}; // Box

static int32_t ReadDataSet(const std::string& data_annotation_file,
                           const std::string& data_root,
                           std::vector<std::string> *all_image_paths,
                           std::vector<std::vector<Box> > *all_gt_boxes) {
  std::string one_line;
  std::string dataset_annotation_str;

  std::ifstream ifile_val_annotation(data_annotation_file);
  if (ifile_val_annotation.is_open()) {
    while (!ifile_val_annotation.eof()) {
      getline(ifile_val_annotation, one_line);
      dataset_annotation_str += one_line + "\n";
    }
  }
  Json::Reader reader;
  JsonValue value_annotation;
  if (!reader.parse(dataset_annotation_str, value_annotation)) {
    cout << "Cannot parse dataset_annotation_str!\n";
    return 0;
  }
  int32_t image_num = value_annotation.size();
  cout << "val image count: " << image_num << endl;

  all_image_paths->assign(image_num, std::string());
  all_gt_boxes->assign(image_num, std::vector<Box>());
  for (int32_t img_idx = 0; img_idx < image_num; ++img_idx) {
    std::string image_path = data_root + value_annotation[img_idx]["filename"].asString();
    (*all_image_paths)[img_idx] = image_path;

    vector<Box> per_img_gt_box;
    JsonValue box_labe_annotation = value_annotation[img_idx]["ann"];
    JsonValue js_bbox = box_labe_annotation["bboxes"];
    JsonValue js_label = box_labe_annotation["labels"];
    CHECK(js_bbox.size() == js_label.size());
    for (int32_t i = 0; i < js_bbox.size(); ++i) {
      float32_t left = js_bbox[i][0].asFloat();
      float32_t top = js_bbox[i][1].asFloat();
      float32_t right = js_bbox[i][2].asFloat();
      float32_t bottom = js_bbox[i][3].asFloat();
      int32_t label = js_label[i].asInt();
      per_img_gt_box.push_back(Box(left, top, right, bottom, label));
    }
    (*all_gt_boxes)[img_idx] = per_img_gt_box;
  }

  return image_num;
}

void eval_recall_at_precision(const vector< vector<Box> > & all_gt_boxes,
                              const vector< vector<Box> > & all_results,
                              const vector< vector<float> > & all_scores,
                              const vector<string> & cls_name,
                              const float prec_val,
                              const float model_thre = 0.0) {
  const float ovthresh = 0.5; // IoU threshold
  const int num_classes = cls_name.size();
  for (int i = 0; i < all_gt_boxes.size(); ++i)
    for (int j = 0; j < all_gt_boxes[i].size(); ++j)
      if (all_gt_boxes[i][j].label >= num_classes) {
        cout << "gt box's label is too large: ";
        cout << all_gt_boxes[i][j].label << " vs " << num_classes;
        cout << endl;
        return;
      }
  for (int i = 0; i < all_results.size(); ++i)
    for (int j = 0; j < all_results[i].size(); ++j)
      if (all_results[i][j].label >= num_classes) {
        cout << "detection result's label is too large: ";
        cout << all_results[i][j].label << " vs " << num_classes;
        cout << endl;
        return;
      }

  const int image_num = all_gt_boxes.size();
  if (all_results.size() != image_num or all_scores.size() != image_num) {
    cout << "image number and result number mismatch" << endl;
    return;
  }
  typedef pair<float, int> P;

  float prec_thd = 0.0;
  float rec_thd = 0.0;
  float thd;
  for (thd = model_thre; thd <= 0.999; thd += 0.001) {
    float tp_thd = 0.0;
    float fp_thd = 0.0;
    float npos_thd = 0.0;
    for (int i = 1; i < num_classes; ++i) {
      const int cur_cls = i;
      vector<Box> results;
      vector<P> scores;
      vector<int> belong_to;

      for (int j = 0; j < image_num; ++j)
        for (int k = 0; k < all_results[j].size(); ++k)
          if (all_results[j][k].label == cur_cls && all_scores[j][k] > thd) {
            belong_to.push_back(j);
            results.push_back(all_results[j][k]);
            scores.push_back(P(all_scores[j][k], scores.size()));
          }
      sort(scores.begin(), scores.end(), std::greater<P>());

      vector<vector<Box> > gt_boxes(image_num, vector<Box>());
      vector<vector<bool> > detected(image_num, vector<bool>());
      for (int j = 0; j < image_num; ++j)
        for (int k = 0; k < all_gt_boxes[j].size(); ++k)
          if (all_gt_boxes[j][k].label == cur_cls) {
            gt_boxes[j].push_back(all_gt_boxes[j][k]);
            detected[j].push_back(false);
            npos_thd += 1.0;
          }

      // calculate fp, tp
      for (int j = 0; j < belong_to.size(); ++j) {
        int result_index = scores[j].second;
        int image_index = belong_to[result_index];
        Box &box = results[result_index];
        vector<Box> &gts = gt_boxes[image_index];
        float ovmax = -1;
        int argmax = -1;
        float box_area = box.area();
        for (int ip = 0; ip < gts.size(); ++ip) {
          Box ovbox = Box(max(gts[ip].x1, box.x1),
                          max(gts[ip].y1, box.y1),
                          min(gts[ip].x2, box.x2),
                          min(gts[ip].y2, box.y2));
          float inter = ovbox.area();
          float suma = max(box_area + gts[ip].area() - inter, (float)EPS);
          float overlap = inter / suma;
          if (overlap > ovmax) {
            ovmax = overlap;
            argmax = ip;
          }
        }
        if (ovmax >= ovthresh && !detected[image_index][argmax]) {
          tp_thd += 1;
          detected[image_index][argmax] = true;
        } else {
          fp_thd += 1;
        }
      }
    }// class

    prec_thd = tp_thd / (tp_thd + fp_thd);
    rec_thd = tp_thd / max(npos_thd, float(EPS));
    if (prec_thd >= prec_val) break;
  }
  cout << "Threshold " << std::fixed << std::setprecision(3) << thd
       << " Precision " << std::fixed << std::setprecision(16) << prec_thd
       << " Recall " << std::fixed << std::setprecision(16) << rec_thd << endl;
}

template <typename Dtype>
void show_det_result(const std::string& img, const Proposal<Dtype> result) {
  cv::Mat
      src = cv::imread(img);
  int length = result.boxes.size();
  for (int i = 0; i < length; ++i) {
    int32_t x_tl = result.boxes[i].x_tl;
    int32_t y_tl = result.boxes[i].y_tl;
    int32_t x_br = result.boxes[i].x_br;
    int32_t y_br = result.boxes[i].y_br;
    int32_t w = x_br - x_tl;
    int32_t h = y_br - y_tl;
    cv::Rect rect(x_tl, y_tl, w, h);
    cv::rectangle(src, rect, cv::Scalar(0, 255, 0));
    std::string showMsg = std::to_string(result.labels[i]) + "|" +
                          std::to_string(result.scores[i]);
    cv::putText(src, showMsg, cv::Point(x_tl, y_tl - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
  }
  cvNamedWindow("finger", CV_WINDOW_AUTOSIZE);
  cv::imshow("finger", src);
  cvWaitKey(0);
}

#endif  // C_OBJECT_FLOW_TEST_H

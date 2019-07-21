#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <cassert>

#include "utils/image_preprocess.h"
#include "test/test.h"

using std::ifstream;
using std::getline;


static void _split(const std::string &s, char delim,
                   std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;

  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  _split(s, delim, elems);
  return elems;
}


int main() {
//  string result_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/data/maskrcnn_cobjectflow_result_py.txt";
  string result_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/data/fasterrcnn_fclayer_cobjectflow_result_py.txt";

  ifstream resf(result_file);
  int32_t result_count = 0;
  string one_line;
  vector<string> lv;
  vector<string> box_str;
  vector<float32_t> box_val;
  vector<vector<float32_t> > box_per_img;
  vector<vector<vector<float32_t> > > box_all;
  float32_t val;
  if (resf.is_open()) {
    while (getline(resf, one_line)) {
//      getline(resf, one_line);
      cout << result_count << " ";
      cout << one_line << endl;
      lv = split(one_line, ';');
      box_per_img = {};

      for (int32_t i = 0; i < lv.size(); ++i) {
        if (lv[i].size() != 1) {
          box_str = split(lv[i], ' ');
          cout << "@ ";
          box_val = {};
          for (int32_t i = 0; i < box_str.size(); ++i) {
            if (box_str[i].size() != 0) {
              std::stringstream stream(box_str[i]);
              stream >> val;
              box_val.push_back(val);
//              cout << std::fixed << std::setprecision(8) << val << "* ";
            }
          }
          assert(box_val.size() == 5);
//          float32_t pc = 1e6;
//          float32_t pc = 1;
          for (int32_t i = 0; i < 4; ++i) {
//            box_val[i] = std::floor(box_val[i] * pc) / pc;
//            box_val[i] = std::floor(box_val[i]);
            cout << std::fixed << std::setprecision(8) << box_val[i] << "* ";
          }
          cout << std::fixed << std::setprecision(8) << box_val[5] << "* ";
          box_per_img.push_back(box_val);
        }
      }
      cout << endl;
      box_all.push_back(box_per_img);
      result_count++;
    }
  }

  cout << "box_all size: " << box_all.size() << endl;
  cout << "result_count: " << result_count << endl;

  string data_root = "data/";
  string dataset_annotation_str;
  ifstream ifile_val_annotation("data/jingtai/VALannotation_ALL.json");
  Json::Reader reader;
  JsonValue value_annotation;

  if (ifile_val_annotation.is_open()) {
    while (!ifile_val_annotation.eof()) {
      getline(ifile_val_annotation, one_line);
      dataset_annotation_str += one_line + "\n";
    }
  }
  if (!reader.parse(dataset_annotation_str, value_annotation)) {
    cout << "Cannot parse dataset_annotation_str!\n";
    return 0;
  }
  int32_t val_image_num = value_annotation.size();
  cout << "val image count: " << val_image_num << endl;

  vector<string> all_image_paths(val_image_num);
  vector<vector<Box> > all_gt_boxes(val_image_num);
  vector<vector<int32_t> > all_image_labels(val_image_num);
  string image_path;
  JsonValue box_labe_annotation;
  JsonValue js_bbox;
  JsonValue js_label;
  vector<Box> per_img_gt_box;
  for (int32_t img_idx = 0; img_idx < val_image_num; ++img_idx) {
    image_path = data_root + value_annotation[img_idx]["filename"].asString();
    all_image_paths[img_idx] = image_path;

    per_img_gt_box = {};
    box_labe_annotation = value_annotation[img_idx]["ann"];
    js_bbox = box_labe_annotation["bboxes"];
    js_label = box_labe_annotation["labels"];
    CHECK(js_bbox.size() == js_label.size());
    for (int32_t i = 0; i < js_bbox.size(); ++i) {
      float32_t left = js_bbox[i][0].asFloat();
      float32_t top = js_bbox[i][1].asFloat();
      float32_t right = js_bbox[i][2].asFloat();
      float32_t bottom = js_bbox[i][3].asFloat();
      int32_t label = js_label[i].asInt();
      per_img_gt_box.push_back(Box(left, top, right, bottom, label));
    }
    all_gt_boxes[img_idx] = per_img_gt_box;
  }

  assert(val_image_num == result_count);
  val_image_num = result_count;
  vector<string> cls_name = { "1", "2" };
  vector<vector<Box> > all_results(val_image_num);
  vector<vector<float> > all_scores(val_image_num);
  vector<Box> results;
  vector<float> scores;

  for (int32_t img_idx = 0; img_idx < val_image_num; ++img_idx) {
    results = {};
    scores = {};
    float32_t score;
    int32_t label;
    vector<vector<float32_t> > box_per_img = box_all[img_idx];
    vector<float32_t> box_val;

    for (int i = 0; i < box_per_img.size(); ++i) {
      box_val = box_per_img[i];
      results.push_back(Box(box_val[0], box_val[1], box_val[2], box_val[3], 1));
      scores.push_back(box_val[4]);
    }
    all_results[img_idx] = results;
    all_scores[img_idx] = scores;
  }

  eval_recall_at_precision(all_gt_boxes, all_results, all_scores, cls_name, 0.99);

}


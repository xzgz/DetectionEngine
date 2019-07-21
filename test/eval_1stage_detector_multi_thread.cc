#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <pthread.h>
//#include <hwloc/helper.h>
#include "detectors/single_stage.h"
#include "cnpy.h"
#include "utils/image_preprocess.h"
#include "test/test.h"

#define NUM_THREADS 56;

//using std::string;
//using std::ifstream;
//using std::getline;
//struct parameter {
//  int32_t thread_id;
//  int32_t thread_num;
//  int32_t picture_num;
//  vector<string> all_image_paths;
//  ImagePreprocess<float32_t> *image_preprocess;
//  vector<int32_t> resize_shape;
//  bool_t flip;
//  bool_t keep_ratio;
//  OneStageDetector<float32_t> *one_stage_detector;
//  vector<vector<Box> > *all_results;
//  vector<vector<float> > *all_scores;
//};
//void* multi_threads_predict(void * arg){
//  auto para = (struct parameter *)arg;
//  int32_t thread_id = para->thread_id;
//  int32_t thread_num = para->thread_num;
//  int32_t picture_num = para->picture_num;
//  vector<string> all_image_paths = para->all_image_paths;
//  ImagePreprocess<float32_t> *image_preprocess = para->image_preprocess;
//  vector<int32_t> resize_shape = para->resize_shape;
//  bool_t flip = para->flip;
//  bool_t keep_ratio = para->keep_ratio;
//  OneStageDetector<float32_t> *two_stage_detector = para->one_stage_detector;
//  vector<vector<Box> > *all_results = para->all_results;
//  vector<vector<float> > *all_scores = para->all_scores;
//  vector<Box> results;
//  vector<float> scores;
//  for (int32_t img_idx = thread_id; img_idx < picture_num; ) {

//    results = {};
//    scores = {};
//    const char * input_image_path = all_image_paths[img_idx].c_str();
//    cout << "img_idx: " << img_idx << "    " << input_image_path << endl;


//    ImgInfo<float32_t> image_info = image_preprocess->ImageTransform(
//        input_image_path, resize_shape, flip, keep_ratio);

//    vector<float32_t*> input_datas(1);
//    vector<int32_t> input_data_counts(1);

//    vector<int32_t> image_rescale_shape = image_info.img_shape;
//    vector<int32_t> image_input_shape = image_info.pad_shape;
//    input_datas[0] = image_info.img;


//    input_data_counts[0] = image_input_shape[0] * image_input_shape[1] * 3;

//    vector<ImageMeta<float32_t> > image_metas(1);
//    ImageMeta<float32_t> image_meta;
//    image_meta.string_shape["image_rescale_shape"] = image_rescale_shape;
//    image_meta.string_shape["image_input_shape"] = image_input_shape;
//    image_meta.string_coefficient["scale_factor"] = image_info.scale_factor[0];
//    image_meta.string_flag["rescale"] = true;
//    image_metas[0] = image_meta;

//    auto proposal_rcnn_list = two_stage_detector->Forward(
//        input_datas, input_data_counts, image_metas);

//    Bbox<float32_t> box;
//    float32_t score;
//    int32_t label;
//    for (int i = 0; i < proposal_rcnn_list[0]->boxes.size(); ++i) {
//      box = proposal_rcnn_list[0]->boxes[i];
//      score = proposal_rcnn_list[0]->scores[i];
//      label = proposal_rcnn_list[0]->labels[i] + 1;
//      results.push_back(Box(box.x_tl, box.y_tl, box.x_br, box.y_br, label));
//      scores.push_back(score);
//    }
//    (*all_results)[img_idx] = results;
//    (*all_scores)[img_idx] = scores;

//    proposal_rcnn_list[0]->PrintBbox();
////    show_det_result<float32_t>(input_image_path, *proposal_rcnn_list[0]);
//    delete image_info.img;
//    img_idx += thread_num;
//  }
//}

int main() {
////  string net_config_file = "examples/retinate_32_bilinear_cfg/retinate_32_bilinear_cfg.txt";
//  string net_config_file = "examples/retinanet_cobjectflow/retinanet_cobjectflow_cfg.txt";

////  cnpy::NpyArray numpy_array = cnpy::npy_load("examples/source1.npy");

//  const char *img_path_finger = "examples/finger.jpg";
//  string data_root = "data/";

//  string one_line;
//  string network_cfg_str;
//  string dataset_annotation_str;

//  ifstream ifile_network_cfg(net_config_file);
//  if (ifile_network_cfg.is_open()) {
//    while (!ifile_network_cfg.eof()) {
//      getline(ifile_network_cfg, one_line);
//      network_cfg_str += one_line + "\n";
//    }
//  }
//  ifstream ifile_val_annotation("data/jingtai/VALannotation_ALL.json");
////  ifstream ifile_val_annotation("data/jingtai/VALannotation_Part.json");
//  if (ifile_val_annotation.is_open()) {
//    while (!ifile_val_annotation.eof()) {
//      getline(ifile_val_annotation, one_line);
//      dataset_annotation_str += one_line + "\n";
//    }
//  }
//  Json::Reader reader;
//  JsonValue value_annotation;
//  JsonValue value_network_cfg;
//  if (!reader.parse(dataset_annotation_str, value_annotation)) {
//    cout << "Cannot parse dataset_annotation_str!\n";
//    return 0;
//  }
//  if (!reader.parse(network_cfg_str, value_network_cfg)) {
//    cout << "Cannot parse network_cfg_str!\n";
//    return 0;
//  }

//  int32_t val_image_num = value_annotation.size();
//  cout << "val image count: " << val_image_num << endl;

//  vector<string> all_image_paths(val_image_num);
//  vector<vector<Box> > all_gt_boxes(val_image_num);
//  vector<vector<int32_t> > all_image_labels(val_image_num);
//  string image_path;
//  JsonValue box_labe_annotation;
//  JsonValue js_bbox;
//  JsonValue js_label;
//  vector<Box> per_img_gt_box;
//  for (int32_t img_idx = 0; img_idx < val_image_num; ++img_idx) {
//    image_path = data_root + value_annotation[img_idx]["filename"].asString();
//    all_image_paths[img_idx] = image_path;

//    per_img_gt_box = {};
//    box_labe_annotation = value_annotation[img_idx]["ann"];
//    js_bbox = box_labe_annotation["bboxes"];
//    js_label = box_labe_annotation["labels"];
//    CHECK(js_bbox.size() == js_label.size());
//    for (int32_t i = 0; i < js_bbox.size(); ++i) {
//      float32_t left = js_bbox[i][0].asFloat();
//      float32_t top = js_bbox[i][1].asFloat();
//      float32_t right = js_bbox[i][2].asFloat();
//      float32_t bottom = js_bbox[i][3].asFloat();
//      int32_t label = js_label[i].asInt();
//      per_img_gt_box.push_back(Box(left, top, right, bottom, label));
//    }
//    all_gt_boxes[img_idx] = per_img_gt_box;
//  }

//  JsonValue net_cfg;
//  JsonValue rcnn_head_cfg;
//  JsonValue rcnn_proposal_cfg;
//  Phase phase = Phase::TEST;
//  net_cfg = value_network_cfg["net_cfg"];
//  rcnn_head_cfg = value_network_cfg["rcnn_head_cfg"];
//  rcnn_proposal_cfg = value_network_cfg["rcnn_proposal_cfg"];

//  OneStageDetector<float32_t> one_stage_detector(
//      net_cfg, phase, rcnn_head_cfg, rcnn_proposal_cfg);
//  vector<shared_ptr<Proposal<float32_t> > > proposal_rcnn_list;

//  vector<float32_t> image_mean = { 103.53, 116.28, 123.675 };
//  vector<float32_t> image_std = { 57.375, 57.12, 58.395 };
////  int32_t size_divisor = 32;
//  int32_t size_divisor = -1;
//  bool_t to_rgb = false;
////  vector<int32_t> resize_shape = { 480, 480 };
//  vector<int32_t> resize_shape = { 240, 320 };
//  bool_t flip = false;
//  bool_t keep_ratio = true;
//  ImagePreprocess<float32_t> *image_preprocess = new ImagePreprocess<float32_t>(
//      image_mean, image_std, size_divisor, to_rgb);

//  vector<string> cls_name = { "1", "2" };
//  vector<vector<Box> > all_results(val_image_num);
//  vector<vector<float> > all_scores(val_image_num);
//  vector<Box> results;
//  vector<float> scores;

//  int32_t num_threads = NUM_THREADS;
//  pthread_t tids[num_threads];
//  parameter *args = new struct parameter[num_threads];
//  for(int t = 0; t < num_threads; ++t)
//  {
//    args[t].all_scores = &all_scores;
//    args[t].all_results = &all_results;
//    args[t].keep_ratio = keep_ratio;
//    args[t].resize_shape = resize_shape;
//    args[t].picture_num = val_image_num;
//    args[t].thread_num = num_threads;
//    args[t].flip = flip;
//    args[t].all_image_paths = all_image_paths;
//    args[t].one_stage_detector = new OneStageDetector<float32_t>(
//        net_cfg, phase, rcnn_head_cfg, rcnn_proposal_cfg);
//    args[t].image_preprocess = image_preprocess;
//    args[t].thread_id = t;
//    pthread_create(&tids[t], NULL, multi_threads_predict, &(args[t]));
//  }
//  for(int t = 0; t < num_threads; ++t)
//    pthread_join(tids[t], NULL);

//  delete image_preprocess;
//  for(int32_t i = 0; i < num_threads; ++i)
//    delete args[i].one_stage_detector;
//  eval_recall_at_precision(all_gt_boxes, all_results, all_scores, cls_name, 0.99);

////  proposal_rcnn_list[0]->PrintBbox();
////  show_det_result<float32_t>(input_image_path, *proposal_rcnn_list[0]);

  return 0;
};

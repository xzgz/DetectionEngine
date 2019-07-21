#include <cmath>
#include <iostream>
#include <fstream>
#include <utility>
#include <pthread.h>

#include "cnpy.h"
#include "inference/inference.h"
#include "test/test.h"

#define NUM_THREADS 4

using std::string;
using std::ifstream;
using std::getline;
struct parameter {
  int32_t thread_id;
  int32_t thread_num;
  int32_t picture_num;
  vector<string> all_image_paths;
  Inference<float32_t> *inference;
  vector<vector<Box> > *all_results;
  vector<vector<float> > *all_scores;
};

void *multi_threads_predict(void *arg){
  auto para = (struct parameter *)arg;
  int32_t thread_id = para->thread_id;
  int32_t thread_num = para->thread_num;
  int32_t picture_num = para->picture_num;
  vector<string> all_image_paths = para->all_image_paths;

  Inference<float32_t> *inference = para->inference;
  vector<vector<Box> > *all_results = para->all_results;
  vector<vector<float> > *all_scores = para->all_scores;
  vector<Box> results;
  vector<float> scores;
  for (int32_t img_idx = thread_id; img_idx < picture_num; ) {
    results = {};
    scores = {};
    string image_path = all_image_paths[img_idx].c_str();

    bool_t rescale = true;
    std::vector<std::shared_ptr<Proposal<float32_t> > > detection_result_list;
    detection_result_list = inference->InferenceSingle(image_path, rescale);

    Bbox<float32_t> box;
    float32_t score;
    int32_t label;
    for (int i = 0; i < detection_result_list[0]->boxes.size(); ++i) {
      box = detection_result_list[0]->boxes[i];
      score = detection_result_list[0]->scores[i];
      label = detection_result_list[0]->labels[i];
      results.push_back(Box(box.x_tl, box.y_tl, box.x_br, box.y_br, label));
      scores.push_back(score);
    }
    (*all_results)[img_idx] = results;
    (*all_scores)[img_idx] = scores;

//    cout << "img_idx: " << img_idx << "    " << image_path << endl;
//    detection_result_list[0]->PrintBbox();
    img_idx += thread_num;
  }
}

void InferenceMultiImage(const vector<string>& image_paths, const string& json_file, const int image_num,
                         const int thread_num, vector<vector<Box> > *results, vector<vector<float> > *scores) {
  pthread_t tids[thread_num];
  parameter *args = new struct parameter[thread_num];
  for(int t = 0; t < thread_num; ++t)
  {
    args[t].all_scores = scores;
    args[t].all_results = results;
    args[t].picture_num = image_num;
    args[t].thread_num = thread_num;
    args[t].all_image_paths = image_paths;
    args[t].inference = new Inference<float32_t>(json_file);
    args[t].thread_id = t;
    pthread_create(&tids[t], 0, multi_threads_predict, &(args[t]));
  }
  for(int t = 0; t < thread_num; ++t)
    pthread_join(tids[t], 0);
  for(int32_t i = 0; i < thread_num; ++i) {
    delete args[i].inference;
  }
  delete [] args;
}

void CreateThread(const vector<string>& image_paths, const string& json_file, const int image_num,
                  pthread_t *thread_id, const int thread_num, vector<vector<Box> > *results,
                  vector<vector<float> > *scores, parameter *args) {
  for(int t = 0; t < thread_num; ++t)
  {
    args[t].all_scores = scores;
    args[t].all_results = results;
    args[t].picture_num = image_num;
    args[t].thread_num = thread_num;
    args[t].all_image_paths = image_paths;
    args[t].inference = new Inference<float32_t>(json_file);
    args[t].thread_id = t;
    pthread_create(&thread_id[t], 0, multi_threads_predict, &(args[t]));
  }
}

void RunThread(pthread_t *thread_id, const int thread_num) {
  for(int t = 0; t < thread_num; ++t) {
    pthread_join(thread_id[t], 0);
  }
}

void DeleteThread(parameter *args, const int thread_num) {
  for(int32_t i = 0; i < thread_num; ++i) {
    delete args[i].inference;
  }
}

int main() {
//  string detection_json_file = "examples/fasterrcnn_fclayer_cobjectflow/roialign/fasterrcnn_fclayer_cobjectflow.txt";
//  string detection_json_file = "examples/mobilenet_cudatest/mobilenet_cudatest.txt";
  string detection_json_file = "examples/maskrcnn_cobjectflow/maskrcnn_cobjectflow.txt";

//  string val_annotation_file = "data/jingtai/VALannotation_ALL.json";
  string val_annotation_file = "data/jingtai/VALannotation_Part.json";
  string data_root = "data/";

  vector<string> all_image_path_temp;
  vector<vector<Box> > all_gt_bbox_temp;
//  int32_t val_image_num = ReadDataSet(val_annotation_file, data_root, &all_image_paths, &all_gt_boxes);
  int image_num = ReadDataSet(val_annotation_file, data_root, &all_image_path_temp, &all_gt_bbox_temp);

  int test_image_num = 400;
  int repeat_count = 10;
  int single_test_num = test_image_num / repeat_count;
//  vector<vector<Box> > all_results(test_image_num);
//  vector<vector<float> > all_scores(test_image_num);
//  InferenceMultiImage(all_image_paths, detection_json_file, test_image_num, NUM_THREADS, &all_results, &all_scores);

//  std::ofstream time_record_file("./inference_time.txt");
  vector<double> inference_times;
  vector<double> create_times;
  pthread_t thread_id[NUM_THREADS];
  parameter *args = new struct parameter[NUM_THREADS];
  for(int i = 0; i < repeat_count; ++i) {
    vector<string> all_image_paths(all_image_path_temp.begin()+i*single_test_num, all_image_path_temp.begin()+(i+1)*single_test_num);
    vector<vector<Box> > all_gt_boxes(all_gt_bbox_temp.begin()+i*single_test_num, all_gt_bbox_temp.begin()+(i+1)*single_test_num);
    cout << "all_image_paths size: " << all_image_paths.size() << endl;
    cout << "all_gt_boxes size: " << all_gt_boxes.size() << endl;
    vector<vector<Box> > all_results(single_test_num);
    vector<vector<float> > all_scores(single_test_num);

    double time = what_time_is_it_now();
    CreateThread(all_image_paths, detection_json_file, single_test_num, thread_id, NUM_THREADS, &all_results, &all_scores, args);
    double create_detector_time = what_time_is_it_now() - time;
    cout << "current time: " << what_time_is_it_now() << endl;
    cout << "create detector time: " << create_detector_time << endl;
    create_times.push_back(create_detector_time);
//    time_record_file << "create detector time: " + std::to_string(create_detector_time) + "\n";
    time = what_time_is_it_now();
    RunThread(thread_id, NUM_THREADS);
    double inference_time = what_time_is_it_now() - time;
    cout << "inference time: " << inference_time << endl;
    inference_times.push_back(inference_time);
//    time_record_file << "inference " + std::to_string(single_test_num) + " images time: " + std::to_string(inference_time) + "\n";
    DeleteThread(args, NUM_THREADS);
  }
  delete [] args;
//  time_record_file.close();

  cout << "create detector time,    inference time,    single_test_num: " << single_test_num << endl;
  double sum_cdt = 0, sum_it = 0;
  for (int i = 0; i < create_times.size(); ++i) {
    cout << create_times[i] << "  " << inference_times[i] << endl;
    sum_cdt += create_times[i];
    sum_it += inference_times[i];
  }
  sum_cdt /= create_times.size();
  sum_it /= create_times.size();
  cout << "create detector average time: " << sum_cdt << " inference average time: " << sum_it
       << " fps: " << single_test_num / sum_it << endl;

//  eval_recall_at_precision(all_gt_boxes, all_results, all_scores, cls_name, 0.99);
//  eval_recall_at_precision(all_gt_boxes, all_results, all_scores, cls_name, 0.99, 0.224);

  return 0;
}

#include <cmath>
#include <utility>

#include "cnpy.h"
#include "inference/inference.h"
#include "test/test.h"

using std::string;
using std::ifstream;
using std::getline;
using namespace cobjectflow;

static void InferenceMultiImage(const vector<string>& image_paths, Inference<float32_t>& inference,
                                vector<vector<Box> > *results, vector<vector<float> > *scores) {
  std::vector<std::shared_ptr<Proposal<float32_t> > > detection_result_list;
  const bool_t rescale = true;
  for (int32_t img_idx = 0; img_idx < image_paths.size(); ++img_idx) {
    vector<Box> result;
    vector<float32_t> score;
    const string& image_path = image_paths[img_idx];
    detection_result_list = inference.InferenceSingle(image_path, rescale);

    Bbox<float32_t> box;
    float32_t s;
    int32_t label;
    for (int i = 0; i < detection_result_list[0]->boxes.size(); ++i) {
      box = detection_result_list[0]->boxes[i];
      s = detection_result_list[0]->scores[i];
      label = detection_result_list[0]->labels[i];
      result.push_back(Box(box.x_tl, box.y_tl, box.x_br, box.y_br, label));
      score.push_back(s);
    }
    (*results)[img_idx] = result;
    (*scores)[img_idx] = score;

//    cout << "img_idx: " << img_idx << "    " << image_path << endl;
//    detection_result_list[0]->PrintBbox();
//    show_det_result<float32_t>(image_path, *detection_result_list[0]);
  }
}

int main() {
//  string detection_json_file = "examples/mobilenetFPN_v2_softmax/mobilenetFPN_v2_softmax_cfg.txt";
//  string detection_json_file = "examples/mobilenet_cudatest/mobilenet_cudatest_cfg.txt";
//  string detection_json_file = "examples/maskrcnn_cobjectflow/maskrcnn_cobjectflow_cfg.txt";

//  string detection_json_file = "examples/fasterrcnn_fclayer_cobjectflow/roialign/fasterrcnn_fclayer_cobjectflow.txt";
//  string detection_json_file = "examples/mobilenet_cudatest/mobilenet_cudatest.txt";
  string detection_json_file = "examples/maskrcnn_cobjectflow/maskrcnn_cobjectflow.txt";

  string val_annotation_file = "data/jingtai/VALannotation_ALL.json";
//  string val_annotation_file = "data/jingtai/VALannotation_RGB.json";
//  string val_annotation_file = "data/jingtai/VALannotation_IR.json";
//  string val_annotation_file = "data/jingtai/VALannotation_Part.json";
  string data_root = "data/";

  vector<string> all_image_path_temp;
  vector<vector<Box> > all_gt_bbox_temp;
  int image_num = ReadDataSet(val_annotation_file, data_root, &all_image_path_temp, &all_gt_bbox_temp);

  int test_image_num = 300;
  int repeat_count = 3;
  int single_test_num = test_image_num / repeat_count;
  vector<vector<Box> > all_results(single_test_num);
  vector<vector<float> > all_scores(single_test_num);

  Inference<float32_t> inference(detection_json_file);
  double cpu_frequency = 3.5e9;
  vector<double> inference_times;
  for(int i = 0; i < repeat_count; ++i) {
    vector<string> all_image_paths(all_image_path_temp.begin()+i*single_test_num, all_image_path_temp.begin()+(i+1)*single_test_num);
    vector<vector<Box> > all_gt_boxes(all_gt_bbox_temp.begin()+i*single_test_num, all_gt_bbox_temp.begin()+(i+1)*single_test_num);
    cout << "all_image_paths size: " << all_image_paths.size() << endl;
    cout << "all_gt_boxes size: " << all_gt_boxes.size() << endl;

    double time = GetCurrentTime(cpu_frequency);
    InferenceMultiImage(all_image_paths, inference, &all_results, &all_scores);
    double inference_time = GetCurrentTime(cpu_frequency) - time;
    cout << "inference time: " << inference_time << "s" << endl;
    inference_times.push_back(inference_time);
  }

  cout << "inference time(s), single_test_num: " << single_test_num << endl;
  double sum_it = 0;
  for (int i = 0; i < inference_times.size(); ++i) {
    cout << inference_times[i] << endl;
    sum_it += inference_times[i];
  }
  sum_it /= inference_times.size();
  cout << " inference average time: " << sum_it << "s"
       << " fps: " << single_test_num / sum_it << endl;

  return 0;
}

////  vector<float32_t> image_mean = { 123.675, 116.28, 103.53 };
////  vector<float32_t> image_std = { 58.395, 57.12, 57.375 };
//  vector<float32_t> image_mean = { 103.53, 116.28, 123.675 };
//  vector<float32_t> image_std = { 57.375, 57.12, 58.395 };
//
////  int32_t size_divisor = 32;
//  int32_t size_divisor = -1;
//
////  bool_t to_rgb = true;
//  bool_t to_rgb = false;
//
////  vector<int32_t> resize_shape = { 480, 480 };
//  vector<int32_t> resize_shape = { 240, 320 };
////  vector<int32_t> resize_shape = { 320, 240 };
//
//  bool_t flip = false;
//  bool_t keep_ratio = true;
//
//  ImagePreprocess<float32_t> *image_preprocess = new ImagePreprocess<float32_t>(
//      image_mean, image_std, size_divisor, to_rgb);
//
//  vector<string> cls_name = { "1", "2" };
//  vector<vector<Box> > all_results(val_image_num);
//  vector<vector<float> > all_scores(val_image_num);
//  vector<Box> results;
//  vector<float> scores;
//  const char *input_image_path;
//  int32_t start_index = 0;
////  int32_t start_index = 3;
//  int32_t end_index = val_image_num;
//  for (int32_t img_idx = start_index; img_idx < end_index; ++img_idx) {
//    results = {};
//    scores = {};
//    input_image_path = all_image_paths[img_idx].c_str();
//    cout << "img_idx: " << img_idx << "    " << input_image_path << endl;
//
////    TwoStageDetector<float32_t> two_stage_detector(first_net_cfg, second_net_cfg,
////        phase, rpn_head_cfg, rpn_proposal_cfg, rcnn_head_cfg, rcnn_proposal_cfg);
////    vector<shared_ptr<Proposal<float32_t> > > proposal_rcnn_list;
//
//    ImgInfo<float32_t> image_info = image_preprocess->ImageTransform(
//        input_image_path, resize_shape, flip, keep_ratio);
//
//    vector<float32_t*> input_datas(1);
//    vector<int32_t> input_data_counts(1);
//
//    vector<int32_t> image_rescale_shape = image_info.img_shape;
//    vector<int32_t> image_input_shape = image_info.pad_shape;
//    input_datas[0] = image_info.img;
//
////    float32_t *raw_data = numpy_array.data<float32_t>();
////    vector<int32_t> image_rescale_shape = { 384, 480 };
////    input_datas[0] = raw_data;
////  input_data_counts[0] = numpy_array.num_vals;
//
//    input_data_counts[0] = image_input_shape[0] * image_input_shape[1] * 3;
//
//    vector<ImageMeta<float32_t> > image_metas(1);
//    ImageMeta<float32_t> image_meta;
//    image_meta.string_shape["image_rescale_shape"] = image_rescale_shape;
//    image_meta.string_shape["image_input_shape"] = image_input_shape;
//    image_meta.string_coefficient["scale_factor"] = image_info.scale_factor[0];
//    image_meta.string_flag["rescale"] = true;
//    image_metas[0] = image_meta;


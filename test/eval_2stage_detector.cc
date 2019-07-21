#include <cmath>
#include <utility>

#include "cnpy.h"
#include "inference/inference.h"
#include "test/test.h"

using std::string;
using std::ifstream;
using std::getline;
using namespace cobjectflow;

int main() {
//  string detection_json_file = "examples/mobilenetFPN_v2_softmax/mobilenetFPN_v2_softmax_cfg.txt";
//  string detection_json_file = "examples/mobilenet_cudatest/mobilenet_cudatest_cfg.txt";
//  string detection_json_file = "examples/maskrcnn_cobjectflow/maskrcnn_cobjectflow_cfg.txt";

//  string detection_json_file = "examples/fasterrcnn_fclayer_cobjectflow/roialign/fasterrcnn_fclayer_cobjectflow.txt";
  string detection_json_file = "examples/mobilenet_cudatest/mobilenet_cudatest.txt";
//  string detection_json_file = "examples/maskrcnn_cobjectflow/maskrcnn_cobjectflow.txt";
//  string val_annotation_file = "data/jingtai/VALannotation_ALL.json";
//  string val_annotation_file = "data/jingtai/VALannotation_RGB.json";
//  string val_annotation_file = "data/jingtai/VALannotation_IR.json";
  string val_annotation_file = "data/jingtai/VALannotation_Part.json";
  string data_root = "data/";

  vector<string> all_image_paths;
  vector<vector<Box> > all_gt_boxes;
  int32_t val_image_num = ReadDataSet(val_annotation_file, data_root, &all_image_paths, &all_gt_boxes);

  Inference<float32_t> inference(detection_json_file);
  bool_t rescale = true;

  std::vector<std::shared_ptr<Proposal<float32_t> > > detection_result_list;
  vector<string> cls_name = {"bg", "hand"};
  vector<vector<Box> > all_results(val_image_num);
  vector<vector<float32_t> > all_scores(val_image_num);
  int32_t start_index = 7;
//  int32_t start_index = 3;
  int32_t end_index = val_image_num;
  for (int32_t img_idx = start_index; img_idx < end_index; ++img_idx) {
    vector<Box> results;
    vector<float32_t> scores;
    const string image_path = all_image_paths[img_idx];
    cout << "img_idx: " << img_idx << "    " << image_path << endl;

    detection_result_list = inference.InferenceSingle(image_path, rescale);

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
    all_results[img_idx] = results;
    all_scores[img_idx] = scores;
    detection_result_list[0]->PrintBbox();
    show_det_result<float32_t>(image_path, *detection_result_list[0]);
  }

  eval_recall_at_precision(all_gt_boxes, all_results, all_scores, cls_name, 0.99);

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


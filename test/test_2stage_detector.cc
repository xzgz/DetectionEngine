//#include <string>
//#include <iostream>
//#include <fstream>
//#include "detectors/two_stage.h"
//#include "cnpy.h"
//#include "utils/image_preprocess.h"
//#include "test/test.h"

//using std::string;
//using std::ifstream;
//using std::getline;


//int main() {
//  string net_config_file = "examples/mobilenetFPN_v2_softmax/mobilenetFPN_v2_softmax_cfg.txt";
//  cnpy::NpyArray numpy_array = cnpy::npy_load("examples/source1.npy");
//  const char *input_image_path = "examples/finger.jpg";

//  string one_line;
//  string all_lines;
//  ifstream ifile(net_config_file);
//  if (ifile.is_open()) {
//    while (!ifile.eof()) {
//      getline(ifile, one_line);
//      all_lines += one_line + "\n";
//    }
//  }
//  cout << "string from file:" << endl;
//  cout << all_lines << endl;

//  const string cfg_str = all_lines;
//  Json::Reader reader;
//  JsonValue value;
//  JsonValue first_net_cfg;
//  JsonValue second_net_cfg;
//  JsonValue rpn_head_cfg;
//  JsonValue rpn_proposal_cfg;
//  JsonValue rcnn_head_cfg;
//  JsonValue rcnn_proposal_cfg;
//  Phase phase = Phase::TEST;
//  if (reader.parse(all_lines, value)) {
//    first_net_cfg = value["first_net_cfg"];
//    second_net_cfg = value["second_net_cfg"];
//    rpn_head_cfg = value["rpn_head_cfg"];
//    rpn_proposal_cfg = value["rpn_proposal_cfg"];
//    rcnn_head_cfg = value["rcnn_head_cfg"];
//    rcnn_proposal_cfg = value["rcnn_proposal_cfg"];
//  }

//  TwoStageDetector<float32_t> two_stage_detector(first_net_cfg, second_net_cfg,
//      phase, rpn_head_cfg, rpn_proposal_cfg, rcnn_head_cfg, rcnn_proposal_cfg);
//  vector<shared_ptr<Proposal<float32_t> > > proposal_rcnn_list;

//  vector<float32_t> image_mean = { 123.675, 116.28, 103.53 };
//  vector<float32_t> image_std = { 58.395, 57.12, 57.375 };
//  int32_t size_divisor = -1;
//  bool_t to_rgb = false;
//  vector<int32_t> resize_shape = { 480, 480 };
//  bool_t flip = false;
//  bool_t keep_ratio = true;
//  ImagePreprocess<float32_t> *image_preprocess = new ImagePreprocess<float32_t>(
//      image_mean, image_std, size_divisor, to_rgb);
//  ImgInfo<float32_t> image_info = image_preprocess->ImageTransform(
//      input_image_path, resize_shape, flip, keep_ratio);

//  vector<float32_t*> input_datas(1);
//  vector<int32_t> input_data_counts(1);

//  vector<int32_t> image_rescale_shape = image_info.img_shape;
//  vector<int32_t> image_input_shape = image_info.pad_shape;
//  input_datas[0] = image_info.img;
//  input_data_counts[0] = image_input_shape[0] * image_input_shape[1] * 3;

////  float32_t *raw_data = numpy_array.data<float32_t>();
////  vector<int32_t> image_rescale_shape = { 384, 480 };
////  input_datas[0] = raw_data;
////  input_data_counts[0] = numpy_array.num_vals;

////  double_t subtract = 0;
////  for(int i = 0; i < 480*480*3 ; ++i){
////      double_t  temp = std::abs(raw_data[i] - image_info.img[i]);
////      if(temp > 0.0001)
////          std::cout << "location: " << i / (480*480) << " " << (i%(480*480)) / 480
////          << " " << (i%(480*480)) % 480 << " loss: " << temp << std::endl;
////      subtract += temp;
////  }
////  std::cout << subtract << std::endl;

//  vector<ImageMeta<float32_t> > image_metas(1);
//  ImageMeta<float32_t> image_meta;
//  image_meta.string_shape["image_rescale_shape"] = image_rescale_shape;
//  image_meta.string_shape["image_input_shape"] = image_input_shape;
//  image_meta.string_coefficient["scale_factor"] = image_info.scale_factor[0];
//  image_meta.string_flag["rescale"] = true;
//  image_metas[0] = image_meta;

//  proposal_rcnn_list = two_stage_detector.Forward(
//      input_datas, input_data_counts, image_metas);

//  show_det_result<float32_t>(input_image_path, *proposal_rcnn_list[0]);

//  return 0;
//};

int main() {

  return 0;
}

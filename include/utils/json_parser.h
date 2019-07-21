#ifndef C_OBJECT_FLOW_JSON_PARSER_H
#define C_OBJECT_FLOW_JSON_PARSER_H

#include "utils/basic_type_def.h"

namespace cobjectflow {

enum InputImageFormat {
  BGR = 0,
  RGB,
  YUV,
  GRAY
};

template <typename Dtype>
struct ImgPreProcessStruct {
  Shape2D scale;
  bool_t to_rgb;
  bool_t flip;
  bool_t resize_keep_ratio;
  int32_t size_divisor;
  InputImageFormat inputimage_format;
  std::vector<Dtype> image_mean;
  std::vector<Dtype> image_std;
};

template <typename Dtype>
struct AnchorGeneratorStruct {
  std::vector<Dtype> anchor_scales;
  std::vector<Dtype> anchor_ratios;
  std::vector<Dtype> anchor_strides;
};

template <typename Dtype>
struct AnchorHeadStruct {
  std::string model_folder;
  int32_t num_cls;
  int32_t nms_pre;
  Dtype iou_thr;
  Dtype min_bbox_size;
  bool_t use_sigmoid_cls;
  bool_t rpn_drop_boxes_runoff_image;
  std::vector<std::vector<std::string> > rcnn_output_name;
  std::vector<Dtype> target_means;
  std::vector<Dtype> target_stds;

  Dtype score_thr;
  int32_t max_per_img;
};

template <typename Dtype>
struct RpnHeadStruct {
  bool_t multi_anchors;
  /// the first stage network folder name
  std::string first_model_folder;
  int32_t nms_pre;
  Dtype nms_thr;
  Dtype min_bbox_size;
  bool_t use_sigmoid_cls;
  bool_t rpn_drop_boxes_runoff_image;
  std::vector<std::vector<std::string> > rpn_network_output_name;
  std::vector<Dtype> target_means;
  std::vector<Dtype> target_stds;

  int32_t nms_post;
  int32_t max_num;
  bool_t nms_each_level;
  bool_t nms_across_levels;
};

template <typename Dtype>
struct RoIExtractorStruct {
  std::string type;
  int32_t out_size;
  int32_t sample_num;
  std::vector<Dtype> featmap_strides;
};

template <typename Dtype>
struct BboxHeadStruct {
  /// the second stage network folder name
  std::string second_model_folder;
  std::vector<std::string> bbox_head_output_name;
  std::vector<std::string> shared_rpn_layer;
  std::vector<Dtype> target_means;
  std::vector<Dtype> target_stds;
  bool_t rpn_drop_boxes_runoff_image;
  Dtype iou_thr;
  Dtype score_thr;
  int32_t max_per_img;
};

template <typename Dtype>
struct DetectionStruct {
  bool_t all_in_one;
  std::string det_type;
  std::shared_ptr<ImgPreProcessStruct<Dtype> > img_preprocess_struct;
  std::shared_ptr<AnchorGeneratorStruct<Dtype> > anchor_generator_struct;
  std::shared_ptr<AnchorHeadStruct<Dtype> > anchor_head_struct;
  std::shared_ptr<RpnHeadStruct<Dtype> > rpn_head_struct;
  std::shared_ptr<RoIExtractorStruct<Dtype> > roi_extractor_struct;
  std::shared_ptr<BboxHeadStruct<Dtype> > bbox_head_struct;
};

template <typename Dtype>
class DetectionJsonParser {
 public:
  DetectionJsonParser(const std::string& json_file);
  std::shared_ptr<DetectionStruct<Dtype> > ParseDetectionStruct();

 private:
  std::shared_ptr<ImgPreProcessStruct<Dtype> > ParseImgPreProcessStruct();
  std::shared_ptr<AnchorGeneratorStruct<Dtype> > ParseAnchorGeneratorStruct();
  std::shared_ptr<AnchorHeadStruct<Dtype> > ParseAnchorHeadStruct();
  std::shared_ptr<RpnHeadStruct<Dtype> > ParseRpnHeadStruct();
  std::shared_ptr<RoIExtractorStruct<Dtype> > ParseRoIExtractorStruct();
  std::shared_ptr<BboxHeadStruct<Dtype> > ParseBboxHeadStruct();
  JsonValue json_value_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_JSON_PARSER_H

#include <fstream>

#include "utils/json_parser.h"

namespace cobjectflow {

template <typename Dtype>
DetectionJsonParser<Dtype>::DetectionJsonParser(const std::string& detection_json_file) {
  std::string one_line;
  std::string json_str = "";
  std::fstream ifile_json_str(detection_json_file);
  if (ifile_json_str.is_open()) {
    while (!ifile_json_str.eof()) {
      std::getline(ifile_json_str, one_line);
      json_str += (one_line + "\n");
    }
  }

  Json::Reader reader;
  if (!reader.parse(json_str, json_value_)) {
    std::cout << "Cannot parse json file!\n";
  }
}

template <typename Dtype>
std::shared_ptr<ImgPreProcessStruct<Dtype> > DetectionJsonParser<Dtype>::ParseImgPreProcessStruct() {
  std::shared_ptr<ImgPreProcessStruct<Dtype> > img_preprocess_struct
  = std::make_shared<ImgPreProcessStruct<Dtype> >();
  EXTRACT_JSON_VALUE_ARRAY_INT32(this->json_value_, scale);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(this->json_value_, image_mean, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(this->json_value_, image_std, Double);
  img_preprocess_struct->scale.h = scale[0];
  img_preprocess_struct->scale.w = scale[1];
  img_preprocess_struct->image_mean = image_mean;
  img_preprocess_struct->image_std = image_std;
  img_preprocess_struct->to_rgb = this->json_value_["to_rgb"].asBool();
  img_preprocess_struct->flip = this->json_value_["flip"].asBool();
  img_preprocess_struct->resize_keep_ratio = this->json_value_["resize_keep_ratio"].asBool();
  img_preprocess_struct->size_divisor = this->json_value_["size_divisor"].asInt();
  std::string image_format = this->json_value_["inputimage_format"].asString();
  if (!image_format.compare("bgr")) {
    img_preprocess_struct->inputimage_format = InputImageFormat::BGR;
  } else if (!image_format.compare("rgb")) {
    img_preprocess_struct->inputimage_format = InputImageFormat::RGB;
  } else if (!image_format.compare("yuv")) {
    img_preprocess_struct->inputimage_format = InputImageFormat::YUV;
  } else if (!image_format.compare("gray")) {
    img_preprocess_struct->inputimage_format = InputImageFormat::GRAY;
  }

  return img_preprocess_struct;
}

template <typename Dtype>
std::shared_ptr<AnchorGeneratorStruct<Dtype> > DetectionJsonParser<Dtype>::ParseAnchorGeneratorStruct() {
  std::shared_ptr<AnchorGeneratorStruct<Dtype> > anchor_generator_struct
  = std::make_shared<AnchorGeneratorStruct<Dtype> >();
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(this->json_value_, anchor_scales, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(this->json_value_, anchor_ratios, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(this->json_value_, anchor_strides, Double);
  anchor_generator_struct->anchor_scales = anchor_scales;
  anchor_generator_struct->anchor_ratios = anchor_ratios;
  anchor_generator_struct->anchor_strides = anchor_strides;

  return anchor_generator_struct;
}

template <typename Dtype>
std::shared_ptr<AnchorHeadStruct<Dtype> > DetectionJsonParser<Dtype>::ParseAnchorHeadStruct() {
  std::shared_ptr<AnchorHeadStruct<Dtype> > anchor_head_struct
  = std::make_shared<AnchorHeadStruct<Dtype> >();
  JsonValue anchor_head = this->json_value_["anchor_head"];
  EXTRACT_JSON_VALUE_ARRAY_STRING_2D(anchor_head, rpn_output_name);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(anchor_head, target_means, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(anchor_head, target_stds, Double);
  anchor_head_struct->rcnn_output_name = rpn_output_name;
  anchor_head_struct->target_means = target_means;
  anchor_head_struct->target_stds = target_stds;

  anchor_head_struct->model_folder = anchor_head["model_folder"].asString();
  anchor_head_struct->use_sigmoid_cls = anchor_head["use_sigmoid_cls"].asBool();
  anchor_head_struct->rpn_drop_boxes_runoff_image = anchor_head["rpn_drop_boxes_runoff_image"].asBool();
  anchor_head_struct->num_cls = anchor_head["num_cls"].asInt();
  anchor_head_struct->nms_pre = anchor_head["nms_pre"].asInt();
  anchor_head_struct->iou_thr = Dtype(anchor_head["iou_thr"].asDouble());
  anchor_head_struct->score_thr = Dtype(anchor_head["score_thr"].asDouble());
  anchor_head_struct->max_per_img = anchor_head["max_per_img"].asInt();
  anchor_head_struct->min_bbox_size = Dtype(anchor_head["min_bbox_size"].asDouble());

  return anchor_head_struct;
}

template <typename Dtype>
std::shared_ptr<RpnHeadStruct<Dtype> > DetectionJsonParser<Dtype>::ParseRpnHeadStruct() {
  std::shared_ptr<RpnHeadStruct<Dtype> > rpn_head_struct
  = std::make_shared<RpnHeadStruct<Dtype> >();
  JsonValue rpn_head = this->json_value_["rpn_head"];
  EXTRACT_JSON_VALUE_ARRAY_STRING_2D(rpn_head, rpn_network_output_name);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(rpn_head, target_means, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(rpn_head, target_stds, Double);
  rpn_head_struct->rpn_network_output_name = rpn_network_output_name;
  rpn_head_struct->target_means = target_means;
  rpn_head_struct->target_stds = target_stds;

  rpn_head_struct->first_model_folder = rpn_head["first_model_folder"].asString();
  rpn_head_struct->nms_pre = rpn_head["nms_pre"].asInt();
  rpn_head_struct->nms_thr = Dtype(rpn_head["nms_thr"].asDouble());
  rpn_head_struct->min_bbox_size = Dtype(rpn_head["min_bbox_size"].asDouble());
  rpn_head_struct->use_sigmoid_cls = rpn_head["use_sigmoid_cls"].asBool();
  rpn_head_struct->rpn_drop_boxes_runoff_image = rpn_head["rpn_drop_boxes_runoff_image"].asBool();

  rpn_head_struct->nms_post = rpn_head["nms_post"].asInt();
  rpn_head_struct->max_num = rpn_head["max_num"].asInt();
  rpn_head_struct->nms_each_level = rpn_head["nms_each_level"].asBool();
  rpn_head_struct->nms_across_levels = rpn_head["nms_across_levels"].asBool();

  return rpn_head_struct;
}

template <typename Dtype>
std::shared_ptr<RoIExtractorStruct<Dtype> > DetectionJsonParser<Dtype>::ParseRoIExtractorStruct() {
  std::shared_ptr<RoIExtractorStruct<Dtype> > roi_extractor_struct
  = std::make_shared<RoIExtractorStruct<Dtype> >();
  JsonValue bbox_roi_extractor = this->json_value_["bbox_roi_extractor"];
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(bbox_roi_extractor, featmap_strides, Double);
  roi_extractor_struct->featmap_strides = featmap_strides;
  roi_extractor_struct->type = bbox_roi_extractor["type"].asString();
  roi_extractor_struct->out_size = bbox_roi_extractor["out_size"].asInt();
  roi_extractor_struct->sample_num = bbox_roi_extractor["sample_num"].asInt();

  return roi_extractor_struct;
}

template <typename Dtype>
std::shared_ptr<BboxHeadStruct<Dtype> > DetectionJsonParser<Dtype>::ParseBboxHeadStruct() {
  std::shared_ptr<BboxHeadStruct<Dtype> > bbox_head_struct
  = std::make_shared<BboxHeadStruct<Dtype> >();
  JsonValue bbox_head = this->json_value_["bbox_head"];
  EXTRACT_JSON_VALUE_ARRAY_STRING(bbox_head, bbox_head_output_name);
  EXTRACT_JSON_VALUE_ARRAY_STRING(bbox_head, shared_rpn_layer);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(bbox_head, target_means, Double);
  EXTRACT_JSON_VALUE_ARRAY_WITH_DTYPE(bbox_head, target_stds, Double);
  bbox_head_struct->bbox_head_output_name = bbox_head_output_name;
  bbox_head_struct->shared_rpn_layer = shared_rpn_layer;
  bbox_head_struct->target_means = target_means;
  bbox_head_struct->target_stds = target_stds;
  bbox_head_struct->second_model_folder = bbox_head["second_model_folder"].asString();
  bbox_head_struct->iou_thr = Dtype(bbox_head["iou_thr"].asDouble());
  bbox_head_struct->score_thr = Dtype(bbox_head["score_thr"].asDouble());
  bbox_head_struct->max_per_img = bbox_head["max_per_img"].asInt();
  bbox_head_struct->rpn_drop_boxes_runoff_image = bbox_head["rpn_drop_boxes_runoff_image"].asBool();

  return bbox_head_struct;
}

template <typename Dtype>
std::shared_ptr<DetectionStruct<Dtype> > DetectionJsonParser<Dtype>::ParseDetectionStruct() {
  std::shared_ptr<DetectionStruct<Dtype> > detection_struct
  = std::make_shared<DetectionStruct<Dtype> >();
  bool_t all_in_one = this->json_value_["all_in_one"].asBool();
  std::string det_type = this->json_value_["det_type"].asString();
  detection_struct->all_in_one = all_in_one;
  detection_struct->det_type = det_type;
  detection_struct->img_preprocess_struct = this->ParseImgPreProcessStruct();
  detection_struct->anchor_generator_struct = this->ParseAnchorGeneratorStruct();
  if (all_in_one) {
    detection_struct->anchor_head_struct = this->ParseAnchorHeadStruct();
  } else {
    detection_struct->rpn_head_struct = this->ParseRpnHeadStruct();
    detection_struct->roi_extractor_struct = this->ParseRoIExtractorStruct();
    detection_struct->bbox_head_struct = this->ParseBboxHeadStruct();
  }

  return detection_struct;
}

template class DetectionJsonParser<float32_t>;

}  // namespace cobjectflow

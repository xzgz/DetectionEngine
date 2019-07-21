#include "inference/inference.h"

namespace cobjectflow {

template <typename Dtype>
Inference<Dtype>::Inference(const std::string& detection_json_file) {
  detection_json_parser_ = new DetectionJsonParser<Dtype>(detection_json_file);
  detection_struct_ = detection_json_parser_->ParseDetectionStruct();
  image_preprocess_ = new ImagePreprocess<Dtype>(
      *(detection_struct_->img_preprocess_struct.get()));

  if (detection_struct_->all_in_one) {
    single_stage_detector_ = new SingleStageDetector<Dtype>(
        *(detection_struct_.get()),
        *(detection_struct_->anchor_generator_struct.get()),
        *(detection_struct_->anchor_head_struct.get()));
  } else {
    two_stage_detector_ = new TwoStageDetector<Dtype>(
        *(detection_struct_.get()),
        *(detection_struct_->anchor_generator_struct.get()),
        *(detection_struct_->rpn_head_struct.get()),
        *(detection_struct_->roi_extractor_struct.get()),
        *(detection_struct_->bbox_head_struct.get()));
  }
}

template <typename Dtype>
Inference<Dtype>::~Inference() {
  delete this->detection_json_parser_;
  delete this->image_preprocess_;
  if (this->detection_struct_->all_in_one) {
    delete this->single_stage_detector_;
  } else {
    delete this->two_stage_detector_;
  }
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<float32_t> > > Inference<Dtype>::InferenceSingle(
    const std::string& image_path, const bool_t rescale) {
  ImgInfo<Dtype> image_info = image_preprocess_->ImageTransform(image_path.c_str());

  std::vector<Dtype*> input_datas(1);
  std::vector<int32_t> input_data_counts(1);
  std::vector<int32_t> image_rescale_shape = image_info.img_shape;
  std::vector<int32_t> image_input_shape = image_info.pad_shape;
  input_datas[0] = image_info.img;
  input_data_counts[0] = image_input_shape[0] * image_input_shape[1] * 3;

  std::vector<ImgMeta<Dtype> > image_metas(1);
  ImgMeta<Dtype> image_meta;
  image_meta.img_shape.c = 3;
  image_meta.img_shape.h = image_rescale_shape[0];
  image_meta.img_shape.w = image_rescale_shape[1];
  image_meta.pad_shape.c = 3;
  image_meta.pad_shape.h = image_input_shape[0];
  image_meta.pad_shape.w = image_input_shape[1];
  image_meta.scale_factor = image_info.scale_factor[0];
  image_metas[0] = image_meta;

  std::vector<std::shared_ptr<Proposal<float32_t> > > proposal_list;
  if (detection_struct_->all_in_one) {
    proposal_list = single_stage_detector_->SimpleTest(input_datas,
        input_data_counts, image_metas, rescale);
  } else {
    std::vector<std::shared_ptr<Proposal<Dtype> > > pre_proposal_list = {};
    proposal_list = two_stage_detector_->SimpleTest(input_datas, input_data_counts,
        image_metas, pre_proposal_list, rescale);
  }

  return proposal_list;
}

template class Inference<float32_t>;

}  // namespace cobjectflow

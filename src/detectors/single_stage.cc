#include "detectors/single_stage.h"

namespace cobjectflow {

template <typename Dtype>
SingleStageDetector<Dtype>::SingleStageDetector(
    const DetectionStruct<Dtype>& detection_struct,
    const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
    const AnchorHeadStruct<Dtype>& anchor_head_struct) {
  std::string proto_txt = anchor_head_struct.model_folder + "/rel.prototxt";
  std::string model_param = anchor_head_struct.model_folder + "/model.bin";
  detector_ = std::make_shared<Detector<Dtype> >(proto_txt, model_param);
  anchor_head_ = std::make_shared<AnchorHead<Dtype> >(
      anchor_generator_struct, anchor_head_struct);
  anchor_head_struct_ = anchor_head_struct;
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<Dtype> > > SingleStageDetector<Dtype>::SimpleTest(
    const std::vector<Dtype*>& input_datas,
    const std::vector<int32_t>& input_data_counts,
    const std::vector<ImgMeta<Dtype> >& img_metas,
    const bool_t rescale) {
  Shape3D image_shape = img_metas[0].pad_shape;
  int32_t image_num = int32_t(img_metas.size());
  const std::vector<int32_t> input_blob_shape =
      { image_num, 3, image_shape.h, image_shape.w };
  this->detector_->ReshapeNet(input_blob_shape);
  this->detector_->SetInputBlobs(input_datas, input_data_counts);
  this->detector_->NetForward();

  /// rcnn stage
  int32_t rcnn_feature_level_num = int32_t(this->anchor_head_struct_.rcnn_output_name.size());
  std::vector<Blob<Dtype>*> cls_scores(rcnn_feature_level_num);
  std::vector<Blob<Dtype>*> bbox_preds(rcnn_feature_level_num);
  for (int32_t i = 0; i < rcnn_feature_level_num; ++i) {
    bbox_preds[i] = this->detector_->blob_by_name(this->anchor_head_struct_.rcnn_output_name[i][1]);
    cls_scores[i] = this->detector_->blob_by_name(this->anchor_head_struct_.rcnn_output_name[i][0]);
  }

  return this->anchor_head_->GetBboxes(cls_scores, bbox_preds, img_metas, rescale);
}

template class SingleStageDetector<float32_t>;

}  // namespace cobjectflow

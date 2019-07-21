#include <memory>

#include "detectors/two_stage.h"
#include "utils/bbox.h"
#include "utils/roi_align.h"

namespace cobjectflow {

template <typename Dtype>
static void Bbox2Roi(const std::vector<Bbox<Dtype> >& bboxes, Data2D<Dtype> *rois) {
  int32_t rois_num = int32_t(bboxes.size());
  for (int32_t i = 0; i < rois_num; ++i) {
    rois->data[rois->shape.w * i + 0] = 0;
    rois->data[rois->shape.w * i + 1] = bboxes[i].x_tl;
    rois->data[rois->shape.w * i + 2] = bboxes[i].y_tl;
    rois->data[rois->shape.w * i + 3] = bboxes[i].x_br;
    rois->data[rois->shape.w * i + 4] = bboxes[i].y_br;
  }
}

template <typename Dtype>
TwoStageDetector<Dtype>::TwoStageDetector(
    const DetectionStruct<Dtype>& detection_struct,
    const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
    const RpnHeadStruct<Dtype>& rpn_head_struct,
    const RoIExtractorStruct<Dtype>& roi_extractor_struct,
    const BboxHeadStruct<Dtype>& bbox_head_struct) {
  std::string proto_txt1 = rpn_head_struct.first_model_folder + "/rel.prototxt";
  std::string model_param1 = rpn_head_struct.first_model_folder + "/model.bin";
  std::string proto_txt2 = bbox_head_struct.second_model_folder + "/rel.prototxt";
  std::string model_param2 = bbox_head_struct.second_model_folder + "/model.bin";
  first_stage_detector_ = std::make_shared<Detector<Dtype> >(proto_txt1, model_param1);
  second_stage_detector_ = std::make_shared<Detector<Dtype> >(proto_txt2, model_param2);
  rpn_head_ = std::make_shared<RpnHead<Dtype> >(
      anchor_generator_struct, rpn_head_struct);
  single_roi_extractor_ = std::make_shared<SingleRoIExtractor<Dtype> >(
      roi_extractor_struct);
  bbox_head_ = std::make_shared<BboxHead<Dtype> >(bbox_head_struct);
  rpn_head_struct_ = rpn_head_struct;
  roi_extractor_struct_ = roi_extractor_struct;
  bbox_head_struct_ = bbox_head_struct;
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<Dtype> > > TwoStageDetector<Dtype>::SimpleTestRpn(
    const std::vector<Dtype*>& input_datas, const std::vector<int32_t>& input_data_counts,
    const std::vector<ImgMeta<Dtype> >& img_metas) {
  int32_t image_num = int32_t(img_metas.size());
  const std::vector<int32_t> input_blob_shape =
      { image_num, 3, img_metas[0].pad_shape.h, img_metas[0].pad_shape.w };
  this->first_stage_detector_->ReshapeNet(input_blob_shape);
  this->first_stage_detector_->SetInputBlobs(input_datas, input_data_counts);
  this->first_stage_detector_->NetForward();

  /// rpn stage
  int32_t rpn_feature_level_num = int32_t(this->rpn_head_struct_.rpn_network_output_name.size());
  std::vector<Blob<Dtype>*> bbox_preds(rpn_feature_level_num);
  std::vector<Blob<Dtype>*> cls_scores(rpn_feature_level_num);
  for (int32_t i = 0; i < rpn_feature_level_num; ++i) {
    bbox_preds[i] = this->first_stage_detector_->blob_by_name(
        this->rpn_head_struct_.rpn_network_output_name[i][1]);
    cls_scores[i] = this->first_stage_detector_->blob_by_name(
        this->rpn_head_struct_.rpn_network_output_name[i][0]);
  }
  std::vector<std::shared_ptr<Proposal<Dtype> > > proposal_rpn_list =
      this->rpn_head_->GetBboxes(cls_scores, bbox_preds, img_metas);

  return proposal_rpn_list;
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<Dtype> > > TwoStageDetector<Dtype>::SimpleTestBboxes(
    const std::vector<ImgMeta<Dtype> >& img_metas,
    const std::vector<std::shared_ptr<Proposal<Dtype> > >& proposal_list,
    bool_t rescale) {
  /// rcnn stage
  int32_t image_num = int32_t(img_metas.size());
  std::vector<std::shared_ptr<Proposal<Dtype> > > proposal_rcnn_list(image_num);
  std::vector<Blob<Dtype>*> rcnn_input_blobs =
      this->first_stage_detector_->blobs_by_names(this->bbox_head_struct_.shared_rpn_layer);
  int32_t rcnn_feature_level_num = int32_t(this->bbox_head_struct_.shared_rpn_layer.size());
  std::vector<Data3D<Dtype> > multi_level_feature(rcnn_feature_level_num);
  Dtype *data_p;
  for (int32_t img_idx = 0; img_idx < image_num; ++img_idx) {
    /// select backbone features of multiple levels to a list for each image
    for (int32_t level = 0; level < rcnn_feature_level_num; ++level) {
      data_p = rcnn_input_blobs[level]->mutable_cpu_data();
      data_p += img_idx * rcnn_input_blobs[level]->count(1);
      multi_level_feature[level].data = data_p;
      multi_level_feature[level].shape.c = rcnn_input_blobs[level]->channels();
      multi_level_feature[level].shape.h = rcnn_input_blobs[level]->height();
      multi_level_feature[level].shape.w = rcnn_input_blobs[level]->width();
    }

    /// extract roi features for each image
    std::shared_ptr<Proposal<Dtype> > proposal_rpn = proposal_list[img_idx];
    int32_t rois_num = int32_t(proposal_rpn->boxes.size());
    Data2D<Dtype> rois;
    rois.shape.h = rois_num;
    rois.shape.w = 5;
    rois.data = new (std::nothrow) Dtype[rois.shape.h * rois.shape.w];
    if (rois.data == NULL) {
      std::cout << "alloc rois.data wrong!" << std::endl;
      return proposal_rcnn_list;
    }
    Bbox2Roi(proposal_rpn->boxes, &rois);
    Data4D<Dtype> roi_features;
    roi_features.shape.n = rois_num;
    roi_features.shape.c = rcnn_input_blobs[0]->channels();
    roi_features.shape.h = this->roi_extractor_struct_.out_size;
    roi_features.shape.w = this->roi_extractor_struct_.out_size;
    roi_features.data = new (std::nothrow) Dtype[
        roi_features.shape.n * roi_features.shape.c *
        roi_features.shape.h * roi_features.shape.w];
    this->single_roi_extractor_->BboxRoIExtractor(multi_level_feature, rois, &roi_features);

    /// pass the roi_features into the second stage network to get the final
    /// predicted bbox deltas and scores
    std::vector<Dtype*> second_stage_input_datas(1);
    std::vector<int32_t> second_stage_input_data_counts(1);
    second_stage_input_datas[0] = roi_features.data;
    second_stage_input_data_counts[0] = roi_features.shape.n
        * roi_features.shape.c * roi_features.shape.h * roi_features.shape.w;
    std::vector<int32_t> roi_features_shape = { roi_features.shape.n, roi_features.shape.c,
                                                roi_features.shape.h, roi_features.shape.w };
    this->second_stage_detector_->ReshapeNet(roi_features_shape);
    this->second_stage_detector_->SetInputBlobs(
        second_stage_input_datas, second_stage_input_data_counts);
    this->second_stage_detector_->NetForward();

    std::vector<Blob<Dtype>*> score_bbox_blobs = this->second_stage_detector_->blobs_by_names(
        this->bbox_head_struct_.bbox_head_output_name);
    Data2D<Dtype> cls_score, bbox_pred;
    cls_score.shape.h = score_bbox_blobs[0]->num();
    cls_score.shape.w = score_bbox_blobs[0]->channels();
    cls_score.data = new Dtype[cls_score.shape.h * cls_score.shape.w];
    memcpy(cls_score.data, score_bbox_blobs[0]->cpu_data(),
        score_bbox_blobs[0]->count(0) * sizeof(Dtype));
    bbox_pred.shape.h = score_bbox_blobs[1]->num();
    bbox_pred.shape.w = score_bbox_blobs[1]->channels();
    bbox_pred.data = new Dtype[bbox_pred.shape.h * bbox_pred.shape.w];
    memcpy(bbox_pred.data, score_bbox_blobs[1]->cpu_data(),
           score_bbox_blobs[1]->count(0) * sizeof(Dtype));
    Shape3D img_shape = img_metas[img_idx].img_shape;
    Dtype scale_factor = img_metas[img_idx].scale_factor;
    std::shared_ptr<Proposal<Dtype> > proposal_rcnn = this->bbox_head_->GetDetBboxes(
        rois, cls_score, bbox_pred, img_shape, scale_factor, rescale);
    proposal_rcnn_list[img_idx] = proposal_rcnn;

    delete [] rois.data;
    delete [] roi_features.data;
    delete [] cls_score.data;
    delete [] bbox_pred.data;
  }

  return proposal_rcnn_list;
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<Dtype> > > TwoStageDetector<Dtype>::SimpleTest(
    const std::vector<Dtype*>& input_datas,
    const std::vector<int32_t>& input_data_counts,
    const std::vector<ImgMeta<Dtype> >& image_metas,
    const std::vector<std::shared_ptr<Proposal<Dtype> > >& proposal_list,
    const bool_t rescale) {
  std::vector<std::shared_ptr<Proposal<Dtype> > > rpn_proposal_list = this->SimpleTestRpn(
      input_datas, input_data_counts, image_metas);
  std::vector<std::shared_ptr<Proposal<Dtype> > > bbox_results = this->SimpleTestBboxes(
      image_metas, rpn_proposal_list, rescale);

  return bbox_results;
}

template class TwoStageDetector<float32_t>;

}  // namespace cobjectflow

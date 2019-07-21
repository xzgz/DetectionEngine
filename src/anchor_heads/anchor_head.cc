#include "anchor_heads/anchor_head.h"
#include "utils/anchor_heads_function.h"

namespace cobjectflow {

template <typename Dtype>
AnchorHead<Dtype>::AnchorHead(
    const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
    const AnchorHeadStruct<Dtype>& anchor_head_struct) {
  anchor_generator_struct_ = anchor_generator_struct;
  anchor_head_struct_ = anchor_head_struct;
  if (anchor_head_struct_.use_sigmoid_cls) {
    cls_out_channels_ = anchor_head_struct_.num_cls - 1;
  } else {
    cls_out_channels_ = anchor_head_struct_.num_cls;
  }
}

template <typename Dtype>
std::vector<std::shared_ptr<Proposal<Dtype> > >& AnchorHead<Dtype>::GetBboxes(
    const std::vector<Blob<Dtype>*>& multi_level_bbox_score_blobs,
    const std::vector<Blob<Dtype>*>& multi_level_bbox_delta_blobs,
    const std::vector<ImgMeta<Dtype> >& img_metas, const bool_t rescale) {
  CHECK(multi_level_bbox_score_blobs.size() == multi_level_bbox_delta_blobs.size() &&
        multi_level_bbox_score_blobs.size() == this->anchor_generator_struct_.anchor_strides.size())
  << ": The length of multi_level_bbox_score_blobs, multi_level_bbox_delta_blobs,"
     " anchor_strides must be the same!";

  this->proposal_list_ = {};
  /// generate anchors for each feature level
  int32_t feature_level_num = int32_t(multi_level_bbox_score_blobs.size());
  std::vector<std::vector<Bbox<Dtype> > > multi_level_anchors(feature_level_num);
  std::vector<int32_t> featmap_size(2);
  for (int32_t i = 0; i < feature_level_num; ++i) {
    featmap_size[0] = multi_level_bbox_score_blobs[i]->height();
    featmap_size[1] = multi_level_bbox_score_blobs[i]->width();
    AnchorGenerator<Dtype> anchor_generator(
        this->anchor_generator_struct_.anchor_strides[i],
        this->anchor_generator_struct_.anchor_scales,
        this->anchor_generator_struct_.anchor_ratios);
    multi_level_anchors[i] = anchor_generator.GenGridAnchors(
        featmap_size, this->anchor_generator_struct_.anchor_strides[i]);
  }

  /// select bbox scores and deltas of multiple feature levels to a list for each image
  int32_t image_num = int32_t(img_metas.size());
  ImgMeta<Dtype> img_meta;
  std::shared_ptr<Proposal<Dtype> > proposal;
  std::vector<Data3D<Dtype> > multi_level_bbox_scores(feature_level_num);
  std::vector<Data3D<Dtype> > multi_level_bbox_deltas(feature_level_num);
  for (int32_t i = 0; i < image_num; ++i) {
    for (int32_t j = 0; j < feature_level_num; ++j) {
      Dtype *data_p = multi_level_bbox_score_blobs[j]->mutable_cpu_data();
      data_p += i * multi_level_bbox_score_blobs[j]->count(1);
      multi_level_bbox_scores[j].data = data_p;
      multi_level_bbox_scores[j].shape.c = multi_level_bbox_score_blobs[j]->channels();
      multi_level_bbox_scores[j].shape.h = multi_level_bbox_score_blobs[j]->height();
      multi_level_bbox_scores[j].shape.w = multi_level_bbox_score_blobs[j]->width();

      data_p = multi_level_bbox_delta_blobs[j]->mutable_cpu_data();
      data_p += i * multi_level_bbox_delta_blobs[j]->count(1);
      multi_level_bbox_deltas[j].data = data_p;
      multi_level_bbox_deltas[j].shape.c = multi_level_bbox_delta_blobs[j]->channels();
      multi_level_bbox_deltas[j].shape.h = multi_level_bbox_delta_blobs[j]->height();
      multi_level_bbox_deltas[j].shape.w = multi_level_bbox_delta_blobs[j]->width();
    }

    img_meta = img_metas[i];
    const Shape3D& img_shape = img_meta.img_shape;
    const Dtype scale_factor = img_meta.scale_factor;
    proposal = this->GetBboxesSingle(multi_level_bbox_scores,
        multi_level_bbox_deltas, multi_level_anchors,
        img_shape, scale_factor, rescale);
    this->proposal_list_.push_back(proposal);
  }
  return this->proposal_list_;
}

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > AnchorHead<Dtype>::GetBboxesSingle(
    const std::vector<Data3D<Dtype> >& multi_level_bbox_scores,
    const std::vector<Data3D<Dtype> >& multi_level_bbox_deltas,
    const std::vector<std::vector<Bbox<Dtype> > >& multi_level_anchors,
    const Shape3D& img_shape, const Dtype scale_factor, const bool_t rescale) {
  /// start process for each feature level
  int32_t feature_level_num = int32_t(multi_level_bbox_scores.size());
  std::vector<std::vector<Dtype> > ml_select_scores;
  std::vector<std::vector<Bbox<Dtype> > > ml_select_bboxes;
  for (int32_t level = 0; level < feature_level_num; ++level) {
    CHECK(multi_level_bbox_scores[level].shape.h == multi_level_bbox_deltas[level].shape.h);
    CHECK(multi_level_bbox_scores[level].shape.w == multi_level_bbox_deltas[level].shape.w);

    /// get the indices sorted by scores in descending order
    std::vector<int32_t> sorted_indices;
    const Data3D<Dtype>& bbox_scores = multi_level_bbox_scores[level];
    int32_t score_size = bbox_scores.shape.c * bbox_scores.shape.h * bbox_scores.shape.w;
    Data2D<Dtype> transposed_scores;
    transposed_scores.data = new Dtype[score_size];
    CalSortedIndex(bbox_scores, &sorted_indices, &transposed_scores,
        this->cls_out_channels_, this->anchor_head_struct_.nms_pre,
        this->anchor_head_struct_.use_sigmoid_cls);

    /// select anchors by indices
    const std::vector<Bbox<Dtype> >& anchors = multi_level_anchors[level];
    std::vector<Bbox<Dtype> > select_anchors(sorted_indices.size());
    SelectAnchors(anchors, sorted_indices, &select_anchors);

    /// select bbox_deltas by indices
    /// at each location, all classes share one bbox of the same ratio and scale
    int32_t bbox_class_num = 1;
    const Data3D<Dtype>& bbox_deltas = multi_level_bbox_deltas[level];
    std::vector<std::vector<BboxDelta<Dtype> > > select_bbox_deltas(sorted_indices.size());
    SelectBboxDeltas(bbox_deltas, sorted_indices, bbox_class_num, &select_bbox_deltas);

    /// select scores by indices and push back to ml_select_scores in each feature level
    SelectScores(transposed_scores, sorted_indices,
        this->anchor_head_struct_.use_sigmoid_cls, &ml_select_scores);
    /// generate refined bboxes with select_anchors and select_bbox_deltas, and
    /// push back to ml_select_bboxes in each feature level
    Delta2Bbox<Dtype>(select_anchors, select_bbox_deltas,
        this->anchor_head_struct_.target_means,
        this->anchor_head_struct_.target_stds,
        img_shape, this->anchor_head_struct_.rpn_drop_boxes_runoff_image,
        &ml_select_bboxes);

    delete [] transposed_scores.data;
  }
  /// rescale the bboxes to the original picture scale
  if (rescale) {
    RescalBboxes(&ml_select_bboxes, scale_factor);
  }

  std::shared_ptr<Proposal<Dtype> > proposal = MulticlassNms<Dtype>(
      ml_select_bboxes, ml_select_scores, this->anchor_head_struct_.score_thr,
      this->anchor_head_struct_.iou_thr, this->anchor_head_struct_.max_per_img);

  return proposal;
}

template class AnchorHead<float32_t>;

}  // namespace cobjectflow

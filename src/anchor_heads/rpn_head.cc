#include "utils/anchor_heads_function.h"
#include "anchor_heads/rpn_head.h"

namespace cobjectflow {

template <typename Dtype>
RpnHead<Dtype>::RpnHead(
    const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
    const RpnHeadStruct<Dtype>& rpn_head_struct)
    : AnchorHead<Dtype>(anchor_generator_struct) {
  rpn_head_struct_ = rpn_head_struct;
  if (rpn_head_struct_.use_sigmoid_cls) this->cls_out_channels_ = 1;
  else this->cls_out_channels_ = 2;
}

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > RpnHead<Dtype>::GetBboxesSingle(
    const std::vector<Data3D<Dtype> >& multi_level_bbox_scores,
    const std::vector<Data3D<Dtype> >& multi_level_bbox_deltas,
    const std::vector<std::vector<Bbox<Dtype> > >& multi_level_anchors,
    const Shape3D& img_shape, const Dtype scale_factor, const bool_t rescale) {
  /// start process for each feature level
  int32_t feature_level_num = int32_t(multi_level_bbox_scores.size());
  std::vector<NMSMetaData<Dtype> > nms_metas;
  for (int32_t level = 0; level < feature_level_num; ++level) {
    CHECK(multi_level_bbox_scores[level].shape.h == multi_level_bbox_deltas[level].shape.h);
    CHECK(multi_level_bbox_scores[level].shape.w == multi_level_bbox_deltas[level].shape.w);
    std::vector<std::vector<Dtype> > select_scores;
    std::vector<std::vector<Bbox<Dtype> > > select_bboxes;
    std::vector<std::vector<Bbox<Dtype> > > valid_bboxes;

    /// get the indices sorted by scores in descending order
    std::vector<int32_t> sorted_indices;
    const Data3D<Dtype>& bbox_scores = multi_level_bbox_scores[level];
    int32_t score_size = bbox_scores.shape.c * bbox_scores.shape.h * bbox_scores.shape.w;
    Data2D<Dtype> transposed_scores;
    transposed_scores.data = new Dtype[score_size];
    CalSortedIndex(bbox_scores, &sorted_indices, &transposed_scores,
        this->cls_out_channels_, this->rpn_head_struct_.nms_pre,
        this->rpn_head_struct_.use_sigmoid_cls);

    /// select anchors by indices
    const std::vector<Bbox<Dtype> >& anchors = multi_level_anchors[level];
    std::vector<Bbox<Dtype> > select_anchors(sorted_indices.size());
    SelectAnchors(anchors, sorted_indices, &select_anchors);

    /// select bbox_deltas by indices
    /// at each location, all classes(bg and fg) share one bbox of the same ratio and scale
    int32_t bbox_class_num = 1;
    const Data3D<Dtype>& bbox_deltas = multi_level_bbox_deltas[level];
    std::vector<std::vector<BboxDelta<Dtype> > > select_bbox_deltas(sorted_indices.size());
    SelectBboxDeltas(bbox_deltas, sorted_indices, bbox_class_num, &select_bbox_deltas);

    /// select bbox_scores by indices
    SelectScores(transposed_scores, sorted_indices,
        this->rpn_head_struct_.use_sigmoid_cls, &select_scores);

    /// generate refined bboxes with select_anchors and select_bbox_deltas
    Delta2Bbox<Dtype>(select_anchors, select_bbox_deltas,
        this->rpn_head_struct_.target_means,
        this->rpn_head_struct_.target_stds,
        img_shape, this->rpn_head_struct_.rpn_drop_boxes_runoff_image,
        &select_bboxes);

    /// drop small bboxes
    DropSmallBboxes(select_bboxes, this->rpn_head_struct_.min_bbox_size, &valid_bboxes);

    /// apply nms on bboxes of each feature level
    NMSEachLevel(select_scores, valid_bboxes, this->rpn_head_struct_.nms_thr,
        this->rpn_head_struct_.nms_post, this->rpn_head_struct_.nms_each_level,
        &nms_metas);
    delete [] transposed_scores.data;
  }
  /// apply nms on boxes of all feature levels
  std::shared_ptr<Proposal<Dtype> > proposal = NMSAcrossLevels(&nms_metas,
      this->rpn_head_struct_.nms_thr, this->rpn_head_struct_.max_num,
      this->rpn_head_struct_.nms_across_levels);

  return proposal;
}

template class RpnHead<float32_t>;

}  // namespace cobjectflow

#include "utils/anchor_heads_function.h"
#include "utils/roi_align.h"
#include "bbox_heads/bbox_head.h"

namespace cobjectflow {

template <typename Dtype>
BboxHead<Dtype>::BboxHead(const BboxHeadStruct<Dtype>& bbox_head_struct) {
  bbox_head_struct_ = bbox_head_struct;
}

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > BboxHead<Dtype>::GetDetBboxes(
    const Data2D<Dtype>& rois, const Data2D<Dtype>& cls_score,
    const Data2D<Dtype>& bbox_pred, const Shape3D& img_shape,
    const Dtype scale_factor, const bool_t rescale) {
  CHECK(rois.shape.w == 5);
  CHECK(bbox_pred.shape.w == cls_score.shape.w * 4);
  CHECK(rois.shape.h == cls_score.shape.h && rois.shape.h == bbox_pred.shape.h);

  int32_t rois_num = rois.shape.h;
  int32_t score_class_num = cls_score.shape.w;
  int32_t bbox_class_num = bbox_pred.shape.w / 4;

  Bbox<Dtype> bbox;
  std::vector<Bbox<Dtype> > refined_anchors(rois_num);
  /// the width of rois is 5, first is batch id, the following 4 values are box coordinates
  for (int32_t i = 0; i < rois_num; ++i) {
    refined_anchors[i] = bbox.CopyFromPointer(rois.data + i * rois.shape.w + 1);
  }

  BboxDelta<Dtype> bbox_delta;
  std::vector<BboxDelta<Dtype> > bbox_deltas_one_class(bbox_class_num - 1);
  std::vector<std::vector<BboxDelta<Dtype> > > bbox_deltas(rois_num);
  for (int32_t i = 0; i < rois_num; ++i) {
    for (int32_t j = 1; j < bbox_class_num; ++j) {
      bbox_deltas_one_class[j - 1] = bbox_delta.CopyFromPointer(
          bbox_pred.data + i * bbox_pred.shape.w + j * 4);
    }
    bbox_deltas[i] = bbox_deltas_one_class;
  }

  std::vector<std::vector<Dtype> > scores(rois_num);
  for (int32_t i = 0; i < rois_num; ++i) {
    std::vector<Dtype> scores_one_class(score_class_num - 1);
    for (int32_t j = 1; j < score_class_num; ++j) {
      scores_one_class[j - 1] = cls_score.data[i * cls_score.shape.w + j];
    }
    scores[i] = scores_one_class;
  }

  std::vector<std::vector<Bbox<Dtype> > > result_bboxes;
  Delta2Bbox<Dtype>(refined_anchors, bbox_deltas,
      this->bbox_head_struct_.target_means,
      this->bbox_head_struct_.target_stds, img_shape,
      this->bbox_head_struct_.rpn_drop_boxes_runoff_image,
      &result_bboxes);

  /// rescale the bboxes to the original picture scale
  if (rescale) {
    RescalBboxes(&result_bboxes, scale_factor);
  }

  std::shared_ptr<Proposal<Dtype> > proposal = MulticlassNms<Dtype>(result_bboxes,
      scores, this->bbox_head_struct_.score_thr, this->bbox_head_struct_.iou_thr,
      this->bbox_head_struct_.max_per_img);

  return proposal;
}

template class BboxHead<float32_t>;

}  // namespace cobjectflow

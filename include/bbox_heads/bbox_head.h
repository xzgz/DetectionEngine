#ifndef C_OBJECT_FLOW_BBOX_HEAD_H
#define C_OBJECT_FLOW_BBOX_HEAD_H

#include "anchor_heads/anchor_head.h"

namespace cobjectflow {

template <typename Dtype>
class BboxHead {
 public:
  BboxHead(const BboxHeadStruct<Dtype>& bbox_head_struct);

  std::shared_ptr<Proposal<Dtype> > GetDetBboxes(
      const Data2D<Dtype>& rois, const Data2D<Dtype>& cls_score,
      const Data2D<Dtype>& bbox_pred, const Shape3D& img_shape,
      const Dtype scale_factor, const bool_t rescale);

 private:
  BboxHeadStruct<Dtype> bbox_head_struct_;
};

}

#endif  // C_OBJECT_FLOW_BBOX_HEAD_H

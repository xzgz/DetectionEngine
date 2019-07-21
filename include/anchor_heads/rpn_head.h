#ifndef C_OBJECT_FLOW_RPN_HEAD_H
#define C_OBJECT_FLOW_RPN_HEAD_H

#include "anchor_heads/anchor_head.h"

namespace cobjectflow {

template <typename Dtype>
class RpnHead : public AnchorHead<Dtype> {
 public:
  RpnHead(
      const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
      const RpnHeadStruct<Dtype>& rpn_head_struct);

  virtual std::shared_ptr<Proposal<Dtype> > GetBboxesSingle(
      const std::vector<Data3D<Dtype> >& multi_level_bbox_scores,
      const std::vector<Data3D<Dtype> >& multi_level_bbox_deltas,
      const std::vector<std::vector<Bbox<Dtype> > >& multi_level_anchors,
      const Shape3D& img_shape, const Dtype scale_factor, const bool_t rescale);

 private:
  RpnHeadStruct<Dtype> rpn_head_struct_;
};

}

#endif  // C_OBJECT_FLOW_RPN_HEAD_H

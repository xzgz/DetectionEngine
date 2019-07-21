#ifndef C_OBJECT_FLOW_ANCHOR_HEAD_H
#define C_OBJECT_FLOW_ANCHOR_HEAD_H

#include <map>
#include <string>

#include "utils/basic_type_def.h"
#include "utils/bbox.h"
#include "utils/nms.h"
#include "utils/math_functions.h"
#include "utils/json_parser.h"
#include "core/anchor/anchor_generator.h"
#include "caffe/blob.hpp"

using caffe::Blob;

namespace cobjectflow {

template <typename Dtype>
class AnchorHead {
 public:
  AnchorHead(
      const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
      const AnchorHeadStruct<Dtype>& anchor_head_struct = AnchorHeadStruct<Dtype>());

  std::vector<std::shared_ptr<Proposal<Dtype> > >& GetBboxes(
      const std::vector<Blob<Dtype>*>& multi_level_bbox_score_blobs,
      const std::vector<Blob<Dtype>*>& multi_level_bbox_delta_blobs,
      const std::vector<ImgMeta<Dtype> >& image_metas,
      const bool_t rescale = false);

  virtual std::shared_ptr<Proposal<Dtype> > GetBboxesSingle(
      const std::vector<Data3D<Dtype> >& multi_level_bbox_scores,
      const std::vector<Data3D<Dtype> >& multi_level_bbox_deltas,
      const std::vector<std::vector<Bbox<Dtype> > >& multi_level_anchors,
      const Shape3D& img_shape, const Dtype scale_factor,
      const bool_t rescale = false);

 protected:
  /// each element corresponds to the point of proposals in one image
  std::vector<std::shared_ptr<Proposal<Dtype> > > proposal_list_;
  AnchorGeneratorStruct<Dtype> anchor_generator_struct_;
  int32_t cls_out_channels_;

 private:
  AnchorHeadStruct<Dtype> anchor_head_struct_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_ANCHOR_HEAD_H

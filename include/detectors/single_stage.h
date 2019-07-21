#ifndef C_OBJECT_FLOW_SINGLE_STAGE_H
#define C_OBJECT_FLOW_SINGLE_STAGE_H

#include "utils/json_parser.h"
#include "detectors/detector.h"
#include "anchor_heads/anchor_head.h"

namespace cobjectflow {

template<typename Dtype>
class SingleStageDetector {
 public:
  SingleStageDetector(
      const DetectionStruct<Dtype>& detection_struct,
      const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
      const AnchorHeadStruct<Dtype>& anchor_head_struct);
  ~SingleStageDetector() {}

  std::vector<std::shared_ptr<Proposal<Dtype> > > SimpleTest(
      const std::vector<Dtype*>& input_datas,
      const std::vector<int32_t>& input_data_counts,
      const std::vector<ImgMeta<Dtype> >& img_metas,
      const bool_t rescale = false);

 private:
  std::shared_ptr<Detector<Dtype> > detector_;
  std::shared_ptr<AnchorHead<Dtype> > anchor_head_;
  AnchorHeadStruct<Dtype> anchor_head_struct_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_SINGLE_STAGE_H

#ifndef C_OBJECT_FLOW_TWO_STAGE_H
#define C_OBJECT_FLOW_TWO_STAGE_H

#include "detectors/detector.h"
#include "anchor_heads/rpn_head.h"
#include "bbox_heads/bbox_head.h"
#include "roi_extractors/single_level.h"

namespace cobjectflow {

template<typename Dtype>
class TwoStageDetector {
 public:
  TwoStageDetector(
      const DetectionStruct<Dtype>& detection_struct,
      const AnchorGeneratorStruct<Dtype>& anchor_generator_struct,
      const RpnHeadStruct<Dtype>& rpn_head_struct,
      const RoIExtractorStruct<Dtype>& roi_extractor_struct,
      const BboxHeadStruct<Dtype>& bbox_head_struct);

  ~TwoStageDetector() {}

  std::vector<std::shared_ptr<Proposal<Dtype> > > SimpleTestRpn(
      const std::vector<Dtype*>& input_datas,
      const std::vector<int32_t>& input_data_counts,
      const std::vector<ImgMeta<Dtype> >& img_metas);

  std::vector<std::shared_ptr<Proposal<Dtype> > > SimpleTestBboxes(
      const std::vector<ImgMeta<Dtype> >& img_metas,
      const std::vector<std::shared_ptr<Proposal<Dtype> > >& proposal_list,
      bool_t rescale);

  std::vector<std::shared_ptr<Proposal<Dtype> > > SimpleTest(
      const std::vector<Dtype*>& input_datas,
      const std::vector<int32_t>& input_data_counts,
      const std::vector<ImgMeta<Dtype> >& image_metas,
      const std::vector<std::shared_ptr<Proposal<Dtype> > >& proposal_list = {},
      const bool_t rescale = false);

 private:
  std::shared_ptr<Detector<Dtype> > first_stage_detector_;
  std::shared_ptr<Detector<Dtype> > second_stage_detector_;
  std::shared_ptr<RpnHead<Dtype> > rpn_head_;
  std::shared_ptr<SingleRoIExtractor<Dtype> > single_roi_extractor_;
  std::shared_ptr<BboxHead<Dtype> > bbox_head_;
  RpnHeadStruct<Dtype> rpn_head_struct_;
  RoIExtractorStruct<Dtype> roi_extractor_struct_;
  BboxHeadStruct<Dtype> bbox_head_struct_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_TWO_STAGE_DETECTOR_H

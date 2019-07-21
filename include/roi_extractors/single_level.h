#ifndef C_OBJECT_FLOW_SINGLE_LEVEL_H
#define C_OBJECT_FLOW_SINGLE_LEVEL_H

#include "utils/basic_type_def.h"
#include "utils/bbox.h"
#include "utils/json_parser.h"
#include "caffe/blob.hpp"

using caffe::Blob;

namespace cobjectflow {

template <typename Dtype>
class SingleRoIExtractor {
 public:
  SingleRoIExtractor(const RoIExtractorStruct<Dtype>& roi_extractor_struct);

  void BboxRoIExtractor(const std::vector<Data3D<Dtype> >& multi_level_feature,
      const Data2D<Dtype>& rois, Data4D<Dtype> *roi_features);

 private:
  int32_t MapRoILevels(const Dtype *rois, int32_t num_levels);
  RoIExtractorStruct<Dtype> roi_extractor_struct_;
  int32_t scale_level_num_;
  Dtype finest_scale_;
};

}

#endif  // C_OBJECT_FLOW_SINGLE_LEVEL_H

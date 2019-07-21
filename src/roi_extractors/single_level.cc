#include "roi_extractors/single_level.h"
#include "utils/roi_align.h"

namespace cobjectflow {

template <typename Dtype>
SingleRoIExtractor<Dtype>::SingleRoIExtractor(
    const RoIExtractorStruct<Dtype>& roi_extractor_struct) {
  roi_extractor_struct_ = roi_extractor_struct;
  scale_level_num_ = int32_t(this->roi_extractor_struct_.featmap_strides.size());
  finest_scale_ = 56.0;
}

template <typename Dtype>
int32_t SingleRoIExtractor<Dtype>::MapRoILevels(
    const Dtype *rois, int32_t num_levels) {
  Dtype scale = std::sqrt((rois[3] - rois[1] + 1) * (rois[4] - rois[2] + 1));
  int32_t level = int32_t(std::floor(std::log2(scale / this->finest_scale_ + 1e-6)));
  (level < 0) ? (level = 0) : ((level > num_levels - 1) ? (level = num_levels - 1) : 0);
  return level;
}

template <typename Dtype>
void SingleRoIExtractor<Dtype>::BboxRoIExtractor(
    const std::vector<Data3D<Dtype> >& multi_level_feature,
    const Data2D<Dtype>& rois, Data4D<Dtype> *roi_features) {
  CHECK(multi_level_feature.size() == this->scale_level_num_);

  std::vector<Dtype> spatial_scales(this->scale_level_num_);
  for (int32_t i = 0; i < this->scale_level_num_; ++i) {
    spatial_scales[i] = Dtype(1.0) / this->roi_extractor_struct_.featmap_strides[i];
  }
  int32_t rois_num = rois.shape.h;
  int32_t rois_width = rois.shape.w;
  int32_t pooled_h = roi_features->shape.h;
  int32_t pooled_w = roi_features->shape.w;
  int32_t each_roi_data_offset
  = roi_features->shape.c * roi_features->shape.h * roi_features->shape.w;
  for (int32_t i = 0; i < rois_num; ++i) {
    int32_t level = this->MapRoILevels(rois.data + rois_width * i, this->scale_level_num_);
    Dtype *top_data = roi_features->data + i * each_roi_data_offset;
    Dtype *roi_box = rois.data + i * rois_width;
    if (!this->roi_extractor_struct_.type.compare("RoIAlign")) {
      RoiAlignForwardCpu<Dtype>(multi_level_feature[level].data,
                                   roi_box,
                                   each_roi_data_offset,
                                   spatial_scales[level],
                                   this->roi_extractor_struct_.sample_num,
                                   multi_level_feature[level].shape.c,
                                   multi_level_feature[level].shape.h,
                                   multi_level_feature[level].shape.w,
                                   pooled_h,
                                   pooled_w,
                                   top_data);
    }
  }
}

template class SingleRoIExtractor<float32_t>;

}  // namespace cobjectflow

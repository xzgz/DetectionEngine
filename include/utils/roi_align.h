#ifndef C_OBJECT_FLOW_ROI_ALIGN_H
#define C_OBJECT_FLOW_ROI_ALIGN_H

#include "utils/basic_type_def.h"
#include "caffe/blob.hpp"

using caffe::Blob;

namespace cobjectflow {

template <typename Dtype>
void RoiAlignForwardCpu(const Dtype *bottom_data, const Dtype *bottom_rois,
                        const int32_t input_data_count, const Dtype spatial_scale,
                        const int32_t sample_num, const int32_t channels,
                        const int32_t height, const int32_t width,
                        const int32_t pooled_height, const int32_t pooled_width,
                        Dtype *top_data);

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_ROI_ALIGN_H

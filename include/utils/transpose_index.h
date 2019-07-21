#ifndef C_OBJECT_FLOW_TRANSPOSE_INDEX_H
#define C_OBJECT_FLOW_TRANSPOSE_INDEX_H

#include "utils/basic_type_def.h"

namespace cobjectflow {

template <typename Dtype>
class TransposeIndex {
 public:
  TransposeIndex(std::vector<int32_t>& array_shape, std::vector<int32_t>& axis_order);
  ~TransposeIndex() {}

  void ResetAxisOrder(std::vector<int32_t> axis_order);
  int64_t offset_at_3dim(int32_t axis1_idx, int32_t axis2_idx, int32_t axis3_idx);

  void TransposeMemory(const Dtype *src, Dtype *dst);

  int32_t array_dim_;
  std::vector<int32_t> axis_order_;
  std::vector<int32_t> origin_shape_;
  std::vector<int32_t> trans_shape_;
  std::vector<int64_t> origin_axis_count_;
  std::vector<int64_t> trans_axis_count_;
  std::vector<int64_t> trans2ori_axis_count_;
};

}  // namespace cobjectflow

#endif //C_OBJECT_FLOW_TRANSPOSE_INDEX_H

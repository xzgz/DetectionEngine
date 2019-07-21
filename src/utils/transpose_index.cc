#include "utils/transpose_index.h"

namespace cobjectflow {

template <typename Dtype>
TransposeIndex<Dtype>::TransposeIndex(std::vector<int32_t>& array_shape,
                                      std::vector<int32_t>& axis_order)
                                      : array_dim_(array_shape.size()),
                                        origin_axis_count_(std::vector<int64_t>(array_dim_)),
                                        trans_axis_count_(std::vector<int64_t>(array_dim_)),
                                        trans2ori_axis_count_(std::vector<int64_t>(array_dim_)),
                                        trans_shape_(array_shape),
                                        origin_shape_(array_shape),
                                        axis_order_(axis_order) {
  CHECK(array_dim_ == axis_order.size()) << ": The array dim must be equal to"
                                            " the length of axis order!";
  int64_t count = 1;
  for (int32_t i = array_dim_ - 1; i >= 0; --i) {
    CHECK_GE(origin_shape_[i], 0);
    CHECK_LE(origin_shape_[i], INT64_MAX / count) << "array size exceeds INT32_MAX";
    origin_axis_count_[i] = count;
    count *= origin_shape_[i];

    trans_shape_[i] = origin_shape_[axis_order_[i]];
  }
  count = 1;
  for (int32_t i = array_dim_ - 1; i >= 0; --i) {
    CHECK_GE(trans_shape_[i], 0);
    CHECK_LE(trans_shape_[i], INT64_MAX / count) << "array size exceeds INT32_MAX";
    trans_axis_count_[i] = count;
    count *= trans_shape_[i];
  }
  for (int32_t i = 0; i < array_dim_; ++i) {
    trans_shape_[i] = origin_shape_[axis_order_[i]];
    trans2ori_axis_count_[i] = origin_axis_count_[axis_order_[i]];
  }
}

template <typename Dtype>
void TransposeIndex<Dtype>::ResetAxisOrder(std::vector<int32_t> axis_order) {
  CHECK(array_dim_ == axis_order.size()) << ": The array dim must be equal to"
                                            " the length of axis order!";
  axis_order_.assign(axis_order.begin(), axis_order.end());
  for (int32_t i = 0; i < array_dim_; ++i) {
    trans_shape_[i] = origin_shape_[axis_order_[i]];
    trans2ori_axis_count_[i] = origin_axis_count_[axis_order_[i]];
  }
  int64_t count = 1;
  for (int32_t i = array_dim_ - 1; i >= 0; --i) {
    CHECK_GE(trans_shape_[i], 0);
    CHECK_LE(trans_shape_[i], INT64_MAX / count) << "array size exceeds INT32_MAX";
    trans_axis_count_[i] = count;
    count *= trans_shape_[i];
  }
}

template <typename Dtype>
void TransposeIndex<Dtype>::TransposeMemory(const Dtype *src, Dtype *dst) {
  for (int32_t i = 0; i < this->trans_shape_[0]; ++i) {
    for (int32_t j = 0; j < this->trans_shape_[1]; ++j) {
      for (int32_t k = 0; k < this->trans_shape_[2]; ++k) {
        dst[i * this->trans_axis_count_[0]
          + j * this->trans_axis_count_[1] + k] =
              src[i * this->trans2ori_axis_count_[0]
                + j * this->trans2ori_axis_count_[1]
                + k * this->trans2ori_axis_count_[2]];
      }
    }
  }
}

template <typename Dtype>
int64_t TransposeIndex<Dtype>::offset_at_3dim(int32_t axis1_idx, int32_t axis2_idx,
                                       int32_t axis3_idx) {
  int32_t axis_idx[3] = {axis1_idx, axis2_idx, axis3_idx};
  int64_t offset = 0;
  for (int32_t i = 0; i < 3; ++i) {
    offset += axis_idx[i] * trans2ori_axis_count_[i];
  }
  return offset;
}

template class TransposeIndex<float32_t>;

}  // namespace cobjectflow

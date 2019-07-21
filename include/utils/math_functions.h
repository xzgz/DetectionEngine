#ifndef C_OBJECT_FLOW_UTIL_MATH_FUNCTIONS_H
#define C_OBJECT_FLOW_UTIL_MATH_FUNCTIONS_H

#include <stdint.h>
#include <cmath>

#include "utils/basic_type_def.h"

namespace cobjectflow {

template <typename Dtype>
struct ValueIndex {
  Dtype   score;
  int32_t index;
};

template <typename Dtype>
inline bool CompareValueIndexGT(const ValueIndex<Dtype>& vi1, const ValueIndex<Dtype>& vi2) {
  return vi1.score > vi2.score;
}

template <typename Dtype>
inline bool CompareValueIndexLT(const ValueIndex<Dtype>& vi1, const ValueIndex<Dtype>& vi2) {
  return vi1.score < vi2.score;
}

template <typename Dtype>
Dtype sigmoid(Dtype x);

template <typename Dtype>
void sigmoid_array_inplace(Data2D<Dtype> *data);

template <typename Dtype>
void sigmoid_array_inplace(Dtype *data, int32_t size);

template <typename Dtype>
void softmax_2dim(Data2D<Dtype> *data, int32_t axis);

template <typename Dtype>
void softmax_2dim(Dtype *data, int32_t rows, int32_t columns, int32_t axis);

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_UTIL_MATH_FUNCTIONS_H

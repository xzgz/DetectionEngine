#include <cfloat>

#include "utils/basic_type_def.h"
#include "utils/math_functions.h"

namespace cobjectflow {

template <typename Dtype>
Dtype sigmoid(Dtype x) {
  return Dtype(1.0) / (Dtype(1.0) + exp(-x));
}

template <typename Dtype>
void sigmoid_array_inplace(Data2D<Dtype> *data) {
  Dtype *data_p = data->data;
  int32_t data_size = data->shape.h * data->shape.w;
  sigmoid_array_inplace<Dtype>(data_p, data_size);
}

template <typename Dtype>
void sigmoid_array_inplace(Dtype *data, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    data[i] = Dtype(1.0) / (Dtype(1.0) + exp(-data[i]));
  }
}

template <typename Dtype>
void softmax_2dim(Data2D<Dtype> *data, int32_t axis) {
  CHECK(axis == 0 || axis == 1);
  Dtype *data_p = data->data;
  int32_t rows = data->shape.h;
  int32_t columns = data->shape.w;
  softmax_2dim<Dtype>(data_p, rows, columns, axis);
}

template <typename Dtype>
void softmax_2dim(Dtype *data, int32_t rows, int32_t columns, int32_t axis) {
  CHECK(axis == 0 || axis == 1);
  Dtype axis_sum;
  Dtype axis_max;
  Dtype temp;
  if (axis == 0) {
    for (int32_t col = 0; col < columns; ++col) {
      axis_max = -FLT_MAX;
      axis_sum = Dtype(0.0);
      for (int32_t row = 0; row < rows; ++row) {
        if (axis_max < data[row * columns + col])
          axis_max = data[row * columns + col];
      }
      for (int32_t row = 0; row < rows; ++row) {
        /// prevent data overflow
        temp = std::exp(data[row * columns + col] - axis_max);
        data[row * columns + col] = temp;
        axis_sum += temp;
      }
      for (int32_t row = 0; row < rows; ++row)
        data[row * columns + col] /= axis_sum;
    }
  } else {
    for (int32_t row = 0; row < rows; ++row) {
      axis_max = -FLT_MAX;
      axis_sum = Dtype(0.0);
      for (int32_t col = 0; col < columns; ++col) {
        if (axis_max < data[row * columns + col])
          axis_max = data[row * columns + col];
      }
      for (int32_t col = 0; col < columns; ++col) {
        /// prevent data overflow
        temp = std::exp(data[row * columns + col] - axis_max);
        data[row * columns + col] = temp;
        axis_sum += temp;
      }
      for (int32_t col = 0; col < columns; ++col) {
        data[row * columns + col] /= axis_sum;
      }
    }
  }
}

template float32_t sigmoid(float32_t x);

template void sigmoid_array_inplace(Data2D<float32_t> *data);
template void sigmoid_array_inplace(float32_t *data, int32_t size);

template void softmax_2dim(Data2D<float32_t> *data, int32_t axis);
template void softmax_2dim(float32_t *data, int32_t rows, int32_t columns, int32_t axis);

}  // namespace cobjectflow

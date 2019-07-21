#include <cmath>
#include <float.h>

#include "utils/roi_align.h"

namespace cobjectflow {

using std::max;
using std::min;

template <typename scalar_t>
scalar_t BilinearInterpolate(const scalar_t *bottom_data,
    const int32_t height, const int32_t width, scalar_t y, scalar_t x) {
  /// deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t) x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  /// do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename T>
void BilinearInterpolatePPL(const T* bottom_data, const int height,
    const int width, T h, T w, T *maxval, int cc) {
  if (h <= 0) h = 0;
  if (w <= 0) w = 0;

  int h_low = (int) h;
  int w_low = (int) w;
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  // do bilinear interpolation
  // get the value of each neighbor
  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  for (int i = 0; i < cc; ++i) {
    T v1 = bottom_data[i * height * width + h_low * width + w_low];
    T v2 = bottom_data[i * height * width + h_low * width + w_high];
    T v3 = bottom_data[i * height * width + h_high * width + w_low];
    T v4 = bottom_data[i * height * width + h_high * width + w_high];

    //get the weight of each neighbor
    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    if (val > maxval[i]) {
      maxval[i] = val;
    }
  }
}

#define SU 1

#if SU

template<typename Dtype>
void RoiAlignForwardCpu(const Dtype *bottom_data, const Dtype *bottom_rois,
    const int32_t input_data_count, const Dtype spatial_scale,
    const int32_t sample_num, const int32_t channels,
    const int32_t height, const int32_t width,
    const int32_t pooled_height, const int32_t pooled_width,
    Dtype *top_data) {
  for (int64_t index = 0; index < input_data_count; ++index) {
    int32_t pw = int32_t(index % pooled_width);
    int32_t ph = int32_t((index / pooled_width) % pooled_height);
    int32_t  c = int32_t((index / pooled_width / pooled_height) % channels);
    int32_t  n = int32_t(index / pooled_width / pooled_height / channels);

    const Dtype *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = (offset_bottom_rois[3] + 1) * spatial_scale;
    Dtype roi_end_h = (offset_bottom_rois[4] + 1) * spatial_scale;
    /// Force malformed ROIs to be 1x1
    Dtype roi_width = std::max(roi_end_w - roi_start_w, Dtype(0.0));
    Dtype roi_height = std::max(roi_end_h - roi_start_h, Dtype(0.0));

    Dtype bin_size_h = roi_height / pooled_height;
    Dtype bin_size_w = roi_width / pooled_width;

    const Dtype *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    int32_t sample_num_h = int32_t((sample_num > 0) ? sample_num
        : std::ceil(roi_height / pooled_height));
    int32_t sample_num_w = int32_t((sample_num > 0) ? sample_num
        : std::ceil(roi_width / pooled_width));

    Dtype h = (Dtype(ph) + Dtype(0.5)) * bin_size_h + roi_start_h;
    Dtype w = (Dtype(pw) + Dtype(0.5)) * bin_size_w + roi_start_w;

    int32_t hstart = std::min(int32_t(std::floor(h)), height - 2);
    int32_t wstart = std::min(int32_t(std::floor(w)), width - 2);

//    Dtype output_val = 0;
//    for (int32_t iy = 0; iy < sample_num_h; iy++) {
//      const Dtype y = roi_start_h + ph * bin_size_h +
//          (Dtype(iy) + Dtype(0.5)) * bin_size_h / (Dtype)(sample_num_h);
//      for (int32_t ix = 0; ix < sample_num_w; ix++) {
//        const Dtype x = roi_start_w + pw * bin_size_w +
//            (Dtype(ix) + Dtype(0.5)) * bin_size_w / (Dtype)(sample_num_w);
//        Dtype val = bilinear_interpolate<Dtype>(offset_bottom_data, height,
//                                                width, y, x);
//        output_val += val;
//      }
//    }
//    output_val /= (sample_num_h * sample_num_w);
//    top_data[index] = output_val;


    Dtype output_val = 0.0;
    Dtype argmax_x = 0.0, argmax_y = 0.0;
    for (int iy = 1; iy <= sample_num_h; iy++) {
      const Dtype y = roi_start_h + ph * bin_size_h +
          (Dtype)(iy) * bin_size_h / (Dtype)(sample_num_h + Dtype(1.0f));
      for (int ix = 1; ix <= sample_num_w; ix++) {
        const Dtype x = roi_start_w + pw * bin_size_w +
            (Dtype)(ix) * bin_size_w / (Dtype)(sample_num_w + Dtype(1.0f));
        Dtype val = BilinearInterpolate<Dtype>(offset_bottom_data, height,
            width, y, x);
        if (val > output_val) {
          output_val = val;
          argmax_x = x;
          argmax_y = y;
        }
      }
    }
    top_data[index] = output_val;
  }
}

#else

template<typename Dtype>
void RoiAlignForwardCpu(Dtype *bottom_data, const Dtype *bottom_rois,
    const int64_t input_data_count, const Dtype spatial_scale,
    const int32_t sample_num, const int32_t channels,
    const int32_t height, const int32_t width,
    const int32_t pooled_height, const int32_t pooled_width,
    Dtype *top_data) {
  int maxn = input_data_count / pooled_width / pooled_height / channels;
  for (int n = 0; n < maxn; ++n) {
    // the n-th roi
    const Dtype *rois = bottom_rois + n * 5;
    //the batch index this roi belongs to
    int roi_batch_ind = rois[0];
    Dtype roi_start_w = rois[1] * spatial_scale;
    Dtype roi_start_h = rois[2] * spatial_scale;
    Dtype roi_end_w = (rois[3] + 1.) * spatial_scale;
    Dtype roi_end_h = (rois[4] + 1.) * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)1.0);
    Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)1.0);
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    // we devide the bin into (sample_num+1)^2 grids, and we sample the inner intersections
    Dtype sample_h = bin_size_h / (sample_num + Dtype(1.0));
    Dtype sample_w = bin_size_w / (sample_num + Dtype(1.0));

    const int c_batch = 40;
    for (int c = 0; c < channels; c += c_batch) {
      int real_c = channels - c > c_batch ? c_batch : channels - c;
      const float *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          // (n, c, ph, pw) is an element in the pooled output
          int index = n * channels * pooled_height * pooled_width +
                      c * pooled_height * pooled_width +
                      ph * pooled_width + pw;

          Dtype hstart = Dtype(ph) * bin_size_h;
          Dtype wstart = Dtype(pw) * bin_size_w;
          Dtype hend = Dtype(ph + 1) * bin_size_h;
          Dtype wend = Dtype(pw + 1) * bin_size_w;

          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart + roi_start_h, Dtype(0)), Dtype(height-1));
          hend = min(max(hend + roi_start_h, Dtype(0)), Dtype(height-1));
          wstart = min(max(wstart + roi_start_w, Dtype(0)), Dtype(width-1));
          wend = min(max(wend + roi_start_w, Dtype(0)), Dtype(width-1));
          bool is_empty = (hend <= hstart) || (wend <= wstart);

          // Define an empty pooling region to be zero
          Dtype maxval[c_batch];
          if (is_empty) {
            for (int i = 0; i < real_c; ++i) {
              maxval[i] = 0;
            }
          } else {
            for (int i = 0; i < real_c; ++i) {
              maxval[i] = 0;
            }
          }
          bool updated = false;
          for (int i = 1; i <= sample_num; ++i) {
            for (int j = 1; j <= sample_num; ++j) {
              Dtype cur_h = hstart + i * sample_h;
              Dtype cur_w = wstart + j * sample_w;
//              if (cur_h >= hend || cur_w >= wend) continue;

              BilinearInterpolatePPL<Dtype>(offset_bottom_data, height, width,
                  cur_h, cur_w, maxval, real_c);
              updated = true;
            }
          }
          for (int cc = 0; cc < real_c; ++cc) {
            top_data[index + cc * pooled_height * pooled_width] = updated ? maxval[cc] : 0;
          }
        }
      }
    }
  }
}

#endif


template float32_t BilinearInterpolate(const float32_t *bottom_data,
    const int32_t height, const int32_t width, float32_t y, float32_t x);

template void BilinearInterpolatePPL(const float32_t *bottom_data,
    const int height, const int width, float32_t h, float32_t w,
    float32_t *maxval, int cc);

template void RoiAlignForwardCpu(const float32_t *bottom_data, const float32_t *bottom_rois,
                                 const int32_t input_data_count, const float32_t spatial_scale,
                                 const int32_t sample_num, const int32_t channels,
                                 const int32_t height, const int32_t width,
                                 const int32_t pooled_height, const int32_t pooled_width,
                                 float32_t *top_data);

}  // namespace cobjectflow

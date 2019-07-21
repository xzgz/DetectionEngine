#include <vector>
#include <cmath>
#include <iomanip>

#include "core/anchor/anchor_generator.h"

namespace cobjectflow {

template <typename Dtype>
AnchorGenerator<Dtype>::AnchorGenerator(const Dtype base_size,
    const std::vector<Dtype>& scales, const std::vector<Dtype>& ratios) {
  base_size_ = base_size;
  scales_ = scales;
  ratios_ = ratios;
}

template <typename Dtype>
void AnchorGenerator<Dtype>::GenBaseAnchors() {
  Dtype x_ctr, y_ctr;
  Dtype w = base_size_;
  Dtype h = base_size_;
  int32_t scale_len = scales_.size();
  int32_t ratio_len = ratios_.size();
  base_anchor_count_ = scale_len * ratio_len;
  Bbox<Dtype> bbox;
  base_anchors_.assign(base_anchor_count_, bbox);
  x_ctr = Dtype(0.5) * (w - 1);
  y_ctr = Dtype(0.5) * (h - 1);
  std::vector<Dtype> h_ratios(ratio_len);
  std::vector<Dtype> w_ratios(ratio_len);

  for (int32_t i = 0; i < ratio_len; ++i) {
    h_ratios[i] = sqrtf(ratios_[i]);
    w_ratios[i] = float32_t(1.0) / h_ratios[i];
  }

  int32_t base_anchor_rows, base_anchor_cols;
  Dtype ws_half, hs_half;
  base_anchor_rows = ratio_len;
  base_anchor_cols = scale_len;
  for (int32_t i = 0; i < base_anchor_rows; ++i) {
    for (int32_t j = 0; j < base_anchor_cols; ++j) {
      ws_half = Dtype(0.5) * (w * w_ratios[i] * scales_[j] - 1);
      hs_half = Dtype(0.5) * (h * h_ratios[i] * scales_[j] - 1);
      base_anchors_[i * base_anchor_cols + j].x_tl = std::round(x_ctr - ws_half);
      base_anchors_[i * base_anchor_cols + j].y_tl = std::round(y_ctr - hs_half);
      base_anchors_[i * base_anchor_cols + j].x_br = std::round(x_ctr + ws_half);
      base_anchors_[i * base_anchor_cols + j].y_br = std::round(y_ctr + hs_half);
    }
  }
}

template <typename Dtype>
std::vector<Bbox<Dtype> >& AnchorGenerator<Dtype>::GenGridAnchors(
    const std::vector<int32_t>& featmap_size, const int32_t stride) {

  this->GenBaseAnchors();

  int32_t feat_h = featmap_size[0];
  int32_t feat_w = featmap_size[1];
  int32_t featw_base_anchor_count = feat_w * base_anchor_count_;
  grid_anchor_count_ = feat_h * feat_w * base_anchor_count_;
  Bbox<Dtype> bbox;
  grid_anchors_.assign(grid_anchor_count_, Bbox<Dtype>());

  for (int32_t i = 0; i < feat_h; ++i) {
    for (int32_t j = 0; j < feat_w; ++j) {
      Dtype img_x = Dtype(j * stride);
      Dtype img_y = Dtype(i * stride);

      for (int32_t k = 0; k < base_anchor_count_; ++k) {
        grid_anchors_[i * featw_base_anchor_count + j * base_anchor_count_ + k].x_tl
        = img_x + base_anchors_[k].x_tl;
        grid_anchors_[i * featw_base_anchor_count + j * base_anchor_count_ + k].y_tl
        = img_y + base_anchors_[k].y_tl;
        grid_anchors_[i * featw_base_anchor_count + j * base_anchor_count_ + k].x_br
        = img_x + base_anchors_[k].x_br;
        grid_anchors_[i * featw_base_anchor_count + j * base_anchor_count_ + k].y_br
        = img_y + base_anchors_[k].y_br;
      }
    }
  }
  return grid_anchors_;
}

template <typename Dtype>
void AnchorGenerator<Dtype>::PrintBaseAnchors() {
  for (int32_t i = 0; i < base_anchor_count_; ++i) {
    std::cout << std::setw(4) << base_anchors_[i].x_tl;
    std::cout << std::setw(4) << base_anchors_[i].y_tl;
    std::cout << std::setw(4) << base_anchors_[i].x_br;
    std::cout << std::setw(4) << base_anchors_[i].y_br << std::endl;
  }
}

template <typename Dtype>
void AnchorGenerator<Dtype>::PrintGridAnchors() {
  for (int32_t i = 0; i < grid_anchor_count_; ++i) {
    std::cout << i << ": ";
    std::cout << std::setw(4) << grid_anchors_[i].x_tl;
    std::cout << std::setw(4) << grid_anchors_[i].y_tl;
    std::cout << std::setw(4) << grid_anchors_[i].x_br;
    std::cout << std::setw(4) << grid_anchors_[i].y_br << std::endl;
  }
}

template class AnchorGenerator<float32_t>;

}  // namespace cobjectflow






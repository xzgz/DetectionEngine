#ifndef C_OBJECT_FLOW_BBOX_H
#define C_OBJECT_FLOW_BBOX_H

#include <cmath>
#include <iostream>
#include <iomanip>

namespace cobjectflow {

template <typename Dtype>
struct Bbox {
  Dtype x_tl;
  Dtype y_tl;
  Dtype x_br;
  Dtype y_br;

  Dtype Area() { return (x_br - x_tl + Dtype(1.0)) * (y_br - y_tl + Dtype(1.0)); }

  Bbox& CopyFromPointer(Dtype *four_element_p) {
    this->x_tl = four_element_p[0];
    this->y_tl = four_element_p[1];
    this->x_br = four_element_p[2];
    this->y_br = four_element_p[3];
    return *this;
  }

  void Floor() {
    this->x_tl = std::floor(this->x_tl);
    this->y_tl = std::floor(this->y_tl);
    this->x_br = std::floor(this->x_br);
    this->y_br = std::floor(this->y_br);
  }

  void Rescale(Dtype scale_factor) {
    this->x_tl *= scale_factor;
    this->y_tl *= scale_factor;
    this->x_br *= scale_factor;
    this->y_br *= scale_factor;
  }

  Bbox& operator = (const Bbox& box) {
    Bbox box_temp;
    this->x_tl = box.x_tl;
    this->y_tl = box.y_tl;
    this->x_br = box.x_br;
    this->y_br = box.y_br;
    return *this;
  }
};

template <typename Dtype>
struct Proposal {
  std::vector<Bbox<Dtype> > boxes;
  std::vector<Dtype>        scores;
  std::vector<int32_t>      labels;
  void PrintBbox() {
    for (int32_t i = 0; i < boxes.size(); ++i) {
      std::cout << std::fixed << std::setprecision(8) << boxes[i].x_tl << "  "
                << std::fixed << std::setprecision(8) << boxes[i].y_tl << "  "
                << std::fixed << std::setprecision(8) << boxes[i].x_br << "  "
                << std::fixed << std::setprecision(8) << boxes[i].y_br << "  "
                << std::fixed << std::setprecision(8) << scores[i] << std::endl;
    }
  }
};

template <typename Dtype>
struct BboxDelta {
  Dtype dx;
  Dtype dy;
  Dtype dw;
  Dtype dh;

  BboxDelta& CopyFromPointer(Dtype *four_element_p) {
    this->dx = four_element_p[0];
    this->dy = four_element_p[1];
    this->dw = four_element_p[2];
    this->dh = four_element_p[3];
    return *this;
  }
};


template <typename Dtype>
void Delta2Bbox(const std::vector<Bbox<Dtype> >& rois,
    const std::vector<std::vector<BboxDelta<Dtype> > >& deltas,
    const std::vector<Dtype>& target_means,
    const std::vector<Dtype>& target_stds,
    const Shape3D& max_shape,
    const bool_t drop_boxes_runoff_image,
    std::vector<std::vector<Bbox<Dtype> > > *result_bboxes,
    const Dtype wh_ratio_clip = Dtype(16.0 / 1000.0));

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_BBOX_H

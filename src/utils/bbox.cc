#include <cmath>

#include "utils/basic_type_def.h"
#include "utils/bbox.h"

namespace cobjectflow {

template <typename Dtype>
static void Delta2BboxSingle(const Bbox<Dtype>& rois,
    const BboxDelta<Dtype>& deltas,
    const Shape3D& max_shape, const Dtype max_ratio,
    const bool_t drop_boxes_runoff_image,
    Bbox<Dtype> *result_bboxes) {
  Bbox<Dtype> bbox;
  Dtype dx, dy, dw, dh;
  Dtype px, py, pw, ph;
  Dtype gx, gy, gw, gh;
  Dtype x_tl, y_tl, x_br, y_br;
  dx = deltas.dx;
  dy = deltas.dy;
  dw = deltas.dw;
  dh = deltas.dh;
  (dw < -max_ratio) ? (dw = -max_ratio) : ((dw > max_ratio) ? (dw = max_ratio) : (0));
  (dh < -max_ratio) ? (dh = -max_ratio) : ((dh > max_ratio) ? (dh = max_ratio) : (0));

  px = (rois.x_tl + rois.x_br) * Dtype(0.5);
  py = (rois.y_tl + rois.y_br) * Dtype(0.5);
  pw = rois.x_br - rois.x_tl + Dtype(1.0);
  ph = rois.y_br - rois.y_tl + Dtype(1.0);
  gx = px + pw * dx;
  gy = py + ph * dy;
  gw = pw * std::exp(dw);
  gh = ph * std::exp(dh);
  x_tl = gx - gw * 0.5 + 0.5;
  y_tl = gy - gh * 0.5 + 0.5;
  x_br = gx + gw * 0.5 - 0.5;
  y_br = gy + gh * 0.5 - 0.5;
  if (drop_boxes_runoff_image) {
    (x_tl < 0) ? (x_tl = 0) : ((x_tl > max_shape.w - 1) ? (x_tl = max_shape.w - 1) : (0));
    (x_br < 0) ? (x_br = 0) : ((x_br > max_shape.w - 1) ? (x_br = max_shape.w - 1) : (0));
    (y_tl < 0) ? (y_tl = 0) : ((y_tl > max_shape.h - 1) ? (y_tl = max_shape.h - 1) : (0));
    (y_br < 0) ? (y_br = 0) : ((y_br > max_shape.h - 1) ? (y_br = max_shape.h - 1) : (0));
  }
  result_bboxes->x_tl = x_tl;
  result_bboxes->y_tl = y_tl;
  result_bboxes->x_br = x_br;
  result_bboxes->y_br = y_br;
}

template <typename Dtype>
void Delta2Bbox(const std::vector<Bbox<Dtype> >& rois,
    const std::vector<std::vector<BboxDelta<Dtype> > >& deltas,
    const std::vector<Dtype>& target_means,
    const std::vector<Dtype>& target_stds,
    const Shape3D& max_shape,
    const bool_t drop_boxes_runoff_image,
    std::vector<std::vector<Bbox<Dtype> > > *result_bboxes,
    const Dtype wh_ratio_clip) {
  CHECK(target_means.size() == 4 && target_stds.size() == 4);

  Dtype max_ratio = std::abs(std::log(wh_ratio_clip));
  int32_t anchor_count = int32_t(rois.size());
  int32_t bbox_class_num = deltas[0].size();
  std::vector<Bbox<Dtype> > bboxes_one_class(bbox_class_num);
  for (int32_t i = 0; i < anchor_count; ++i) {
    for (int32_t j = 0; j < bbox_class_num; ++j) {
      BboxDelta<Dtype> bbox_delta;
      Bbox<Dtype> result_bbox;
      bbox_delta.dx = deltas[i][j].dx * target_stds[0] + target_means[0];
      bbox_delta.dy = deltas[i][j].dy * target_stds[1] + target_means[1];
      bbox_delta.dw = deltas[i][j].dw * target_stds[2] + target_means[2];
      bbox_delta.dh = deltas[i][j].dh * target_stds[3] + target_means[3];
      Delta2BboxSingle(rois[i], bbox_delta, max_shape, max_ratio,
          drop_boxes_runoff_image, &result_bbox);
      bboxes_one_class[j] = result_bbox;
    }
    result_bboxes->push_back(bboxes_one_class);
  }
}

template void Delta2Bbox(const std::vector<Bbox<float32_t> >& rois,
    const std::vector<std::vector<BboxDelta<float32_t> > >& deltas,
    const std::vector<float32_t>& target_means,
    const std::vector<float32_t>& target_stds,
    const Shape3D& max_shape,
    const bool_t drop_boxes_runoff_image,
    std::vector<std::vector<Bbox<float32_t> > > *result_bboxes,
    const float32_t wh_ratio_clip);

}  // cobjectflow

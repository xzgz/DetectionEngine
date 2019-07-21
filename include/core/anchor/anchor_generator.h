#ifndef C_OBJECT_FLOW_ANCHOR_GENERATOR_H
#define C_OBJECT_FLOW_ANCHOR_GENERATOR_H

#include <cmath>

#include "utils/basic_type_def.h"
#include "utils/bbox.h"

namespace cobjectflow {

template <typename Dtype>
class AnchorGenerator {
 public:
  AnchorGenerator(const Dtype base_size, const std::vector<Dtype>& scales,
      const std::vector<Dtype>& ratios);

  void GenBaseAnchors();
  std::vector<Bbox<Dtype> >& GenGridAnchors(
      const std::vector<int32_t>& featmap_size, const int32_t stride);

  void PrintBaseAnchors();
  void PrintBaseAnchorSize();
  void PrintGridAnchors();
  void PrintGridAnchorSize();

  int32_t base_anchor_count() { return base_anchor_count_; }
  int32_t grid_anchor_count() { return grid_anchor_count_; }

  std::vector<Bbox<Dtype> >& base_anchors() { return base_anchors_; }
  std::vector<Bbox<Dtype> >& grid_anchors() { return grid_anchors_; }

 private:
  Dtype base_size_;
  std::vector<Dtype> scales_;
  std::vector<Dtype> ratios_;
  int32_t base_anchor_count_;
  int32_t grid_anchor_count_;
  std::vector<Bbox<Dtype> > base_anchors_;
  std::vector<Bbox<Dtype> > grid_anchors_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_ANCHOR_GENERATOR_H

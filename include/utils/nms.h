#ifndef C_OBJECT_FLOW_NMS_H
#define C_OBJECT_FLOW_NMS_H

#include "utils/basic_type_def.h"
#include "bbox.h"

namespace cobjectflow {

template <typename Dtype>
struct NMSMetaData {
  Bbox<Dtype> bbox;
  Dtype score;
  int32_t label;
  int32_t index;
};

template <typename Dtype>
inline bool CompareNMSMetaDataGT(const NMSMetaData<Dtype>& v1,
    const NMSMetaData<Dtype>& v2) {
  return v1.score > v2.score;
}

template <typename Dtype>
inline bool CompareNMSMetaDataLT(const NMSMetaData<Dtype>& v1,
    const NMSMetaData<Dtype>& v2) {
  return v1.score < v2.score;
}


template <typename Dtype>
Dtype IoU(Bbox<Dtype>& box_a, Bbox<Dtype>& box_b);

template <typename Dtype>
void NmsCpu(std::vector<NMSMetaData<Dtype> > *nms_metas, Dtype iou_thr,
    bool_t is_sort, std::vector<int32_t> *keeped_indices);

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > MulticlassNms(
    const std::vector<std::vector<Bbox<Dtype> > >& ml_select_bboxes,
    const std::vector<std::vector<Dtype> >& ml_select_scores,
    const Dtype score_thr, const Dtype iou_thr, const int32_t max_num);

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_NMS_H

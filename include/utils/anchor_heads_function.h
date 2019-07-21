#ifndef C_OBJECT_FLOW_ANCHOR_HEAD_FUNCTION_H
#define C_OBJECT_FLOW_ANCHOR_HEAD_FUNCTION_H

#include "utils/basic_type_def.h"
#include "math_functions.h"
#include "utils/bbox.h"
#include "utils/nms.h"

namespace cobjectflow {

template <typename Dtype>
void NormaliseScores(Data2D<Dtype> *scores, const bool_t use_sigmoid_cls);

template <typename Dtype>
void InitializeScoreIndex(ValueIndex<Dtype> *scores_indices,
    const Data2D<Dtype> *scores, const bool_t use_sigmoid_cls);

template <typename Dtype>
void CalSortedIndex(const Data3D<Dtype>& bbox_scores,
    std::vector<int32_t> *sorted_index,
    Data2D<Dtype> *transposed_scores,
    const int32_t cls_out_channels,
    const int32_t nms_pre,
    const bool_t use_sigmoid_cls);

template <typename Dtype>
void SelectAnchors(const std::vector<Bbox<Dtype> >& anchors,
    const std::vector<int32_t>& sorted_index,
    std::vector<Bbox<Dtype> > *select_anchors);

template <typename Dtype>
void SelectBboxDeltas(const Data3D<Dtype>& bbox_deltas,
    const std::vector<int32_t>& sorted_index, const int32_t bbox_class_num,
    std::vector<std::vector<BboxDelta<Dtype> > > *select_bbox_deltas);

template <typename Dtype>
void SelectScores(const Data2D<Dtype>& scores,
    const std::vector<int32_t>& sorted_index, const bool_t use_sigmoid_cls,
    std::vector<std::vector<Dtype> > *select_scores);

template <typename Dtype>
void DropSmallBboxes(const std::vector<std::vector<Bbox<Dtype> > >& bboxes,
    const Dtype min_bbox_size, std::vector<std::vector<Bbox<Dtype> > > *valid_bboxes);

template <typename Dtype>
void RescalBboxes(std::vector<std::vector<Bbox<Dtype> > > *bboxes,
    const Dtype scale_factor);

template <typename Dtype>
void NMSEachLevel(const std::vector<std::vector<Dtype> >& scores,
    const std::vector<std::vector<Bbox<Dtype> > >& bboxes, const Dtype nms_thr,
    const int32_t nms_post, const bool_t nms_each_level,
    std::vector<NMSMetaData<Dtype> > *nms_metas);

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > NMSAcrossLevels(
    std::vector<NMSMetaData<Dtype> > *nms_metas, const Dtype nms_thr,
    const int32_t max_num, const bool_t nms_across_levels);

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_ANCHOR_HEAD_FUNCTION_H

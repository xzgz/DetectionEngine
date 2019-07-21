#include <cfloat>

#include "utils/math_functions.h"
#include "utils/transpose_index.h"
#include "utils/anchor_heads_function.h"

namespace cobjectflow {

template <typename Dtype>
void NormaliseScores(Data2D<Dtype> *scores, const bool_t use_sigmoid_cls) {
  if (use_sigmoid_cls) {
    sigmoid_array_inplace<Dtype>(scores);
  } else {
    int32_t axis = 1;
    softmax_2dim<Dtype>(scores, axis);
  }
}

template <typename Dtype>
void InitializeScoreIndex(ValueIndex<Dtype> *scores_indices,
    const Data2D<Dtype> *scores, const bool_t use_sigmoid_cls) {
  int32_t scores_size = scores->shape.h * scores->shape.w;
  /// to ensure the scores of small feature map to be arranged in front
  for (int32_t i = 0; i < scores_size; ++i) {
    scores_indices[i].score = -FLT_MAX;
    scores_indices[i].index = i;
  }
  int32_t trans_score_h = scores->shape.h;
  int32_t trans_score_w = scores->shape.w;
  int32_t cls_start_idx = (use_sigmoid_cls) ? (0) : (1);
  for (int32_t i = 0; i < trans_score_h; ++i) {
    Dtype class_score_max = Dtype(-1.0);
    for (int32_t j = cls_start_idx; j < trans_score_w; ++j) {
      Dtype score_temp = scores->data[i * trans_score_w + j];
      if (class_score_max < score_temp) {
        class_score_max = score_temp;
      }
    }
    scores_indices[i].score = class_score_max;
    scores_indices[i].index = i;
  }
}

template <typename Dtype>
void CalSortedIndex(const Data3D<Dtype>& bbox_scores,
    std::vector<int32_t> *sorted_index,
    Data2D<Dtype> *transposed_scores,
    const int32_t cls_out_channels,
    const int32_t nms_pre,
    const bool_t use_sigmoid_cls) {
  /// get transposed scores
  std::vector<int32_t> axis_order = { 1, 2, 0 };
  std::vector<int32_t> scores_shape = {
      bbox_scores.shape.c, bbox_scores.shape.h, bbox_scores.shape.w };
  TransposeIndex<Dtype> trans_index(scores_shape, axis_order);
  trans_index.TransposeMemory(bbox_scores.data, transposed_scores->data);
  /// reshape the transposed score to 2-D, the lower dim stores the score of each class
  int32_t scores_size = bbox_scores.shape.c * bbox_scores.shape.h * bbox_scores.shape.w;
  transposed_scores->shape.w = cls_out_channels;
  transposed_scores->shape.h = scores_size / transposed_scores->shape.w;
  NormaliseScores<Dtype>(transposed_scores, use_sigmoid_cls);
  /// get score_index and use it for sorting
  ValueIndex<Dtype> *score_index = new ValueIndex<Dtype>[scores_size];
  InitializeScoreIndex<Dtype>(score_index, transposed_scores, use_sigmoid_cls);
  /// sort score_index
  int32_t select_anchor_count = std::min(nms_pre, transposed_scores->shape.h);
  std::partial_sort(score_index, score_index + select_anchor_count,
                    score_index + transposed_scores->shape.h, CompareValueIndexGT<Dtype>);
//  std::stable_sort(score_index, score_index + score_shape_new[0], CompareValueIndexGT<Dtype>);
//  std::sort(score_index, score_index + score_shape_new[0], CompareValueIndexGT<Dtype>);
  sorted_index->assign(select_anchor_count, 0);
  for (int32_t i = 0; i < select_anchor_count; ++i) {
    (*sorted_index)[i] = score_index[i].index;
  }

  delete [] score_index;
}

template <typename Dtype>
void SelectAnchors(const std::vector<Bbox<Dtype> >& anchors,
    const std::vector<int32_t>& sorted_index,
    std::vector<Bbox<Dtype> > *select_anchors) {
  int32_t select_anchor_count = int32_t(sorted_index.size());
  for (int32_t i = 0; i < select_anchor_count; ++i) {
    (*select_anchors)[i] = anchors[sorted_index[i]];
  }
}

template <typename Dtype>
void SelectBboxDeltas(const Data3D<Dtype>& bbox_deltas,
    const std::vector<int32_t>& sorted_index, const int32_t bbox_class_num,
    std::vector<std::vector<BboxDelta<Dtype> > > *select_bbox_deltas) {
  int32_t select_bbox_count = int32_t(sorted_index.size());
  int32_t bbox_delta_hw = bbox_deltas.shape.h * bbox_deltas.shape.w;
  int32_t high_addr_offset = 4 * bbox_delta_hw;
  int32_t base_anchor_count = bbox_deltas.shape.c / 4;
  std::vector<BboxDelta<Dtype> > one_class_deltas(bbox_class_num);
  Bbox<Dtype> bbox;
  for (int32_t i = 0; i < select_bbox_count; ++i) {
    int32_t high_addr = sorted_index[i] % base_anchor_count;
    int32_t low_addr = sorted_index[i] / base_anchor_count;
    int32_t base_addr = high_addr * high_addr_offset + low_addr;
    for (int32_t j = 0; j < bbox_class_num; ++j) {
      one_class_deltas[j].dx = bbox_deltas.data[base_addr + (j * 4 + 0) * bbox_delta_hw];
      one_class_deltas[j].dy = bbox_deltas.data[base_addr + (j * 4 + 1) * bbox_delta_hw];
      one_class_deltas[j].dw = bbox_deltas.data[base_addr + (j * 4 + 2) * bbox_delta_hw];
      one_class_deltas[j].dh = bbox_deltas.data[base_addr + (j * 4 + 3) * bbox_delta_hw];
    }
    (*select_bbox_deltas)[i] = one_class_deltas;
  }
}

template <typename Dtype>
void SelectScores(const Data2D<Dtype>& scores,
    const std::vector<int32_t>& sorted_index, const bool_t use_sigmoid_cls,
    std::vector<std::vector<Dtype> > *select_scores) {
  int32_t select_score_count = int32_t(sorted_index.size());
  int32_t score_w = scores.shape.w;
  std::vector<Dtype> one_class_scores(score_w);
  int32_t class_start_index = (use_sigmoid_cls) ? (0) : (1);
  for (int32_t i = 0; i < select_score_count; ++i) {
    for (int32_t j = 0; j < score_w; ++j) {
      int32_t addr = sorted_index[i] * score_w + class_start_index + j;
      one_class_scores[j] = scores.data[addr];
    }
    select_scores->push_back(one_class_scores);
  }
}

template <typename Dtype>
void DropSmallBboxes(const std::vector<std::vector<Bbox<Dtype> > >& bboxes,
    const Dtype min_bbox_size, std::vector<std::vector<Bbox<Dtype> > > *valid_bboxes) {
  CHECK(bboxes[0].size() == 1);

  int32_t bbox_count = int32_t(bboxes.size());
  std::vector<Bbox<Dtype> > bboxes_one_class(1);
  for (int32_t i = 0; i < bbox_count; ++i) {
    Dtype w = bboxes[i][0].x_br - bboxes[i][0].x_tl + 1;
    Dtype h = bboxes[i][0].y_br - bboxes[i][0].y_tl + 1;
    if (min_bbox_size > 0.0f) {
      if (w >= min_bbox_size && h >= min_bbox_size) {
        bboxes_one_class[0] = bboxes[i][0];
        valid_bboxes->push_back(bboxes_one_class);
      }
    } else {
      bboxes_one_class[0] = bboxes[i][0];
      valid_bboxes->push_back(bboxes_one_class);
    }
  }
}

template <typename Dtype>
void RescalBboxes(std::vector<std::vector<Bbox<Dtype> > > *bboxes,
    const Dtype scale_factor) {
  for (int32_t i = 0; i < bboxes->size(); ++i) {
    for (int32_t j = 0; j < (*bboxes)[i].size(); ++j) {
      (*bboxes)[i][j].Rescale(Dtype(1.0) / scale_factor);
    }
  }
}

template <typename Dtype>
void NMSEachLevel(const std::vector<std::vector<Dtype> >& scores,
    const std::vector<std::vector<Bbox<Dtype> > >& bboxes, const Dtype nms_thr,
    const int32_t nms_post, const bool_t nms_each_level,
    std::vector<NMSMetaData<Dtype> > *nms_metas) {
  CHECK(scores.size() == bboxes.size());
  int32_t bboxes_count = int32_t(bboxes.size());
  NMSMetaData<Dtype> nms_meta_temp;
  if (nms_each_level) {
    std::vector<NMSMetaData<Dtype> > nms_metas_temp(bboxes_count);
    std::vector<int32_t> keeped_indices;
    for (int32_t i = 0; i < bboxes_count; ++i) {
      nms_metas_temp[i].bbox = bboxes[i][0];
      nms_metas_temp[i].score = scores[i][0];
    }
    bool_t is_sort = false;
    NmsCpu<Dtype>(&nms_metas_temp, nms_thr, is_sort, &keeped_indices);
    int32_t valid_bboxes_count = std::min(int32_t(keeped_indices.size()), nms_post);
    for (int32_t i = 0; i < valid_bboxes_count; ++i) {
      nms_metas->push_back(nms_metas_temp[keeped_indices[i]]);
    }
  } else {
    for (int32_t i = 0; i < bboxes_count; ++i) {
      nms_meta_temp.bbox = bboxes[i][0];
      nms_meta_temp.score = scores[i][0];
      nms_metas->push_back(nms_meta_temp);
    }
  }
}

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > NMSAcrossLevels(
    std::vector<NMSMetaData<Dtype> > *nms_metas, const Dtype nms_thr,
    const int32_t max_num, const bool_t nms_across_levels) {
  std::sort(nms_metas->begin(), nms_metas->end(), CompareNMSMetaDataGT<Dtype>);
  int32_t metas_count = int32_t(nms_metas->size());
  std::vector<int32_t> keeped_indices;
  if (nms_across_levels) {
    bool_t is_sort = false;
    NmsCpu<Dtype>(nms_metas, nms_thr, is_sort, &keeped_indices);
  } else {
    keeped_indices.assign(metas_count, 0);
    for (int32_t i = 0; i < metas_count; ++i) {
      keeped_indices[i] = i;
    }
  }
  int32_t bboxes_count = std::min(int32_t(keeped_indices.size()), max_num);
  std::shared_ptr<Proposal<Dtype> > proposal(new Proposal<Dtype>());
  /// we only use the box coordinates in the following steps of two-stage detection algorithm
  proposal->boxes.assign(bboxes_count, Bbox<Dtype>());
  for (int32_t i = 0; i < bboxes_count; ++i) {
    proposal->boxes[i] = (*nms_metas)[keeped_indices[i]].bbox;
  }

  return proposal;
}

template void NormaliseScores(Data2D<float32_t> *scores, const bool_t use_sigmoid_cls);

template void InitializeScoreIndex(ValueIndex<float32_t> *scores_indices,
                          const Data2D<float32_t> *scores, const bool_t use_sigmoid_cls);

template void CalSortedIndex(const Data3D<float32_t>& bbox_scores,
                    std::vector<int32_t> *sorted_index,
                    Data2D<float32_t> *transposed_scores,
                    const int32_t cls_out_channels,
                    const int32_t nms_pre,
                    const bool_t use_sigmoid_cls);

template void SelectAnchors(const std::vector<Bbox<float32_t> >& anchors,
                   const std::vector<int32_t>& sorted_index,
                   std::vector<Bbox<float32_t> > *select_anchors);

template void SelectBboxDeltas(const Data3D<float32_t>& bbox_deltas,
                      const std::vector<int32_t>& sorted_index, const int32_t bbox_class_num,
                      std::vector<std::vector<BboxDelta<float32_t> > > *select_bbox_deltas);

template void SelectScores(const Data2D<float32_t>& scores,
                  const std::vector<int32_t>& sorted_index, const bool_t use_sigmoid_cls,
                  std::vector<std::vector<float32_t> > *select_scores);

template void DropSmallBboxes(const std::vector<std::vector<Bbox<float32_t> > >& bboxes,
                     const float32_t min_bbox_size, std::vector<std::vector<Bbox<float32_t> > > *valid_bboxes);

template void RescalBboxes(std::vector<std::vector<Bbox<float32_t> > > *bboxes,
                  const float32_t scale_factor);

template void NMSEachLevel(const std::vector<std::vector<float32_t> >& scores,
                  const std::vector<std::vector<Bbox<float32_t> > >& bboxes, const float32_t nms_thr,
                  const int32_t nms_post, const bool_t nms_each_level,
                  std::vector<NMSMetaData<float32_t> > *nms_metas);

template std::shared_ptr<Proposal<float32_t> > NMSAcrossLevels(
    std::vector<NMSMetaData<float32_t> > *nms_metas, const float32_t nms_thr,
    const int32_t max_num, const bool_t nms_across_levels);

}  // namespace cobjectflow

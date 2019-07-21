#include <cmath>
#include <algorithm>

#include "utils/nms.h"

namespace cobjectflow {

template <typename Dtype>
Dtype IoU(Bbox<Dtype>& box_a, Bbox<Dtype>& box_b) {
  Dtype xx1 = std::max(box_a.x_tl, box_b.x_tl);
  Dtype yy1 = std::max(box_a.y_tl, box_b.y_tl);
  Dtype xx2 = std::min(box_a.x_br, box_b.x_br);
  Dtype yy2 = std::min(box_a.y_br, box_b.y_br);
  Dtype inter = std::max(Dtype(0.0), xx2 - xx1 + Dtype(1.0))
      * std::max(Dtype(0.0), yy2 - yy1 + Dtype(1.0));
  return inter / (box_a.Area() + box_b.Area() - inter);
}

template <typename Dtype>
void NmsCpu(std::vector<NMSMetaData<Dtype> > *nms_metas, Dtype iou_thr,
    bool_t is_sort, std::vector<int32_t> *keeped_indices) {
  int32_t data_count = int32_t(nms_metas->size());
  std::vector<bool_t> suppressed(data_count, false);
  Bbox<Dtype> ibox, jbox;
  if (is_sort) {
    std::sort(nms_metas->begin(), nms_metas->end(), CompareNMSMetaDataGT<Dtype>);
  }

  int32_t select_index;
  Dtype ovr;
  for (int32_t i = 0; i < data_count; ++i) {
    if (suppressed[i]) continue;
    if (is_sort) {
      select_index = (*nms_metas)[i].index;
    } else {
      select_index = i;
    }
    keeped_indices->push_back(select_index);
    ibox = (*nms_metas)[i].bbox;
    for (int32_t j = i + 1; j < data_count; ++j) {
      jbox = (*nms_metas)[j].bbox;
      ovr = IoU<Dtype>(ibox, jbox);
      if (ovr >= iou_thr) suppressed[j] = true;
    }
  }
}

template <typename Dtype>
std::shared_ptr<Proposal<Dtype> > MulticlassNms(
    const std::vector<std::vector<Bbox<Dtype> > >& bboxes,
    const std::vector<std::vector<Dtype> >& scores,
    const Dtype score_thr, const Dtype iou_thr,
    const int32_t max_num) {
  int32_t bbox_class_num = bboxes[0].size();
  int32_t score_class_num = scores[0].size();
  if (bbox_class_num != 1) {
    CHECK(bbox_class_num == score_class_num);
  }
  CHECK(bboxes.size() == scores.size());

  /// apply nms for each class
  std::shared_ptr<Proposal<Dtype> > proposals(new Proposal<Dtype>());
  int32_t bbox_count = int32_t(bboxes.size());
  std::vector<std::vector<int32_t> > post_nms_index(score_class_num);
  int32_t post_nms_count = 0;
  bool_t is_sort = true;
  for (int cls = 0; cls < score_class_num; ++cls) {
    std::vector<NMSMetaData<Dtype> > nms_metas;
    NMSMetaData<Dtype> nms_meta;
    for (int32_t bc = 0; bc < bbox_count; ++bc) {
      Dtype score = scores[bc][cls];
      if (score > score_thr) {
        nms_meta.score = score;
        nms_meta.index = bc;
        if (bbox_class_num == 1) {
          nms_meta.bbox = bboxes[bc][0];
        } else {
          nms_meta.bbox = bboxes[bc][cls];
        }
        nms_metas.push_back(nms_meta);
      }
    }
    std::vector<int32_t> keeped_indices;
    NmsCpu<Dtype>(&nms_metas, iou_thr, is_sort, &keeped_indices);
    post_nms_index[cls] = keeped_indices;
    post_nms_count += post_nms_index[cls].size();
  }

  /// merge the result bboxes of all classes to a list, then select top-max_num
  /// bboxes according to scores in descending order
  int32_t bbox_result_count = std::min(post_nms_count, max_num);
  Bbox<Dtype> result_bbox;
  int32_t count = 0;
  std::vector<NMSMetaData<Dtype> > post_nms_results(post_nms_count);
  for (int32_t cls = 0; cls < score_class_num; ++cls) {
    for (int32_t i = 0; i < post_nms_index[cls].size(); ++i) {
      int32_t select_index = post_nms_index[cls][i];
      post_nms_results[count].index = select_index;
      post_nms_results[count].label = cls;
      post_nms_results[count].score = scores[select_index][cls];
      count++;
    }
  }
  if (post_nms_count > max_num) {
    std::partial_sort(post_nms_results.begin(), post_nms_results.begin() + max_num,
        post_nms_results.end(), CompareNMSMetaDataGT<Dtype>);
  }

  for (int32_t i = 0; i < bbox_result_count; ++i) {
    int32_t select_index = post_nms_results[i].index;
    int32_t result_label = post_nms_results[i].label;
    Dtype result_score = scores[select_index][result_label];

//    float32_t pc = 1e1;
    float32_t pc = 1;
    if (bbox_class_num == 1) {
      result_bbox = bboxes[select_index][0];
    } else {
      result_bbox = bboxes[select_index][result_label];
    }
//    result_bbox.Rescale(pc);
//    result_bbox.Floor();
//    result_bbox.Rescale(Dtype(1.0) / pc);

    proposals->boxes.push_back(result_bbox);
    proposals->scores.push_back(result_score);
    proposals->labels.push_back(result_label + 1);
  }
  return proposals;
}

template float32_t IoU(Bbox<float32_t>& box_a, Bbox<float32_t>& box_b);

template void NmsCpu(std::vector<NMSMetaData<float32_t> > *nms_metas,
    float32_t iou_thr, bool_t is_sort, std::vector<int32_t> *keeped_indices);

template std::shared_ptr<Proposal<float32_t> > MulticlassNms(
    const std::vector<std::vector<Bbox<float32_t> > >& ml_select_bboxes,
    const std::vector<std::vector<float32_t> >& ml_select_scores,
    const float32_t score_thr, const float32_t iou_thr, const int32_t max_num);

}  // namespace cobjectflow

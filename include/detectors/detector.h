// this file will be replaced by NNLib.
#ifndef C_OBJECT_FLOW_DETECTOR_H
#define C_OBJECT_FLOW_DETECTOR_H

#include "utils/basic_type_def.h"
#include "utils/json_parser.h"
#include "caffe/blob.hpp"
#include "caffe/net.hpp"

using caffe::Blob;
using caffe::Net;
using caffe::Phase;

namespace cobjectflow {

template <typename Dtype>
class Detector {
 public:
  Detector(
      const std::string& proto_txt, const std::string& model_param,
      const Phase phase = Phase::TEST);

  void NetForward();
  void SetInputBlobs(
      const std::vector<Dtype*>& input_datas,
      const std::vector<int32_t>& input_data_counts);
  void ReshapeNet(
      const std::vector<int32_t>& shape);

  const std::vector<Blob<Dtype>*>& output_feature_blobs();
  Blob<Dtype> *blob_by_name(const std::string& blob_name);
  const std::vector<Blob<Dtype>*> blobs_by_names(
      const std::vector<std::string>& output_blob_names);

  void PrintOneBlob(const std::string& blob_name);
  void PrintOneBlob(const Blob<Dtype> *blob);

  JsonValue net_cfg_;

 private:
  std::shared_ptr<Net<Dtype> > net_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_DETECTOR_H

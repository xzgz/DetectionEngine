#ifndef C_OBJECT_FLOW_INFERENCE_H
#define C_OBJECT_FLOW_INFERENCE_H

#include "utils/json_parser.h"
#include "utils/image_preprocess.h"
#include "detectors/single_stage.h"
#include "detectors/two_stage.h"

namespace cobjectflow {

template <typename Dtype>
class Inference {
 public:
  Inference(const std::string& detection_json_file);
  ~Inference();

  std::vector<std::shared_ptr<Proposal<float32_t> > > InferenceSingle(
      const std::string& image_path, const bool_t rescale);

 private:
  DetectionJsonParser<Dtype> *detection_json_parser_;
  ImagePreprocess<Dtype> *image_preprocess_;
  SingleStageDetector<Dtype> *single_stage_detector_;
  TwoStageDetector<Dtype> *two_stage_detector_;
  std::shared_ptr<DetectionStruct<Dtype> > detection_struct_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_INFERENCE_H

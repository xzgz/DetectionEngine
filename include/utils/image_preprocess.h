#ifndef C_OBJECT_FLOW_IMAGE_PREPROCESS_H
#define C_OBJECT_FLOW_IMAGE_PREPROCESS_H

#include "basic_type_def.h"
#include "bbox.h"
#include "utils/json_parser.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace cobjectflow {

template <typename Dtype>
class ImagePreprocess {
 public:
  ImagePreprocess(const ImgPreProcessStruct<Dtype>& img_preprocess_struct);

  ImgInfo<Dtype> ImageTransform(const std::string& path);

 private:
  cv::Mat img;
  ImgPreProcessStruct<Dtype> img_preprocess_struct_;
};

}  // namespace cobjectflow

#endif  // C_OBJECT_FLOW_IMAGE_PREPROCESS_H

#include "utils/image_preprocess.h"

using std::vector;

namespace cobjectflow {

template <typename Dtype>
inline vector<int32_t > _scale_size(int32_t h, int32_t w, Dtype scale_factor){
  return vector<int32_t >{int32_t(w * float32_t(scale_factor) + 0.5),
                          int32_t(h * float32_t(scale_factor) + 0.5)};
}

vector<float32_t> imresize(cv::Mat& img, vector<int32_t> new_size, int interpolation=cv::INTER_LINEAR){
  cv::Size size = cv::Size(new_size[0], new_size[1]);
  cv::resize(img, img, size);
  return vector<float32_t >{float32_t(new_size[0]) / float32_t(img.size().width), float32_t(new_size[1]) / float32_t(img.size().height)};
}

vector<float32_t > imrescale(cv::Mat& img, vector<int32_t> scale){
  int32_t h, w;
  vector<int32_t > new_size;
  float32_t scale_factor;
  h = img.rows;
  w = img.cols;
  if (scale.size() == 1) {
    CHECK(scale[0] > 0);
    scale_factor = scale[0];
  } else {
    CHECK(scale.size() == 2);
    float32_t max_long_edge = std::max(scale[0], scale[1]);
    float32_t min_long_edge = std::min(scale[0], scale[1]);

    scale_factor = float32_t(std::min(max_long_edge / std::max(h, w), min_long_edge / std::min(h, w)));
  }
  new_size = _scale_size(h, w, scale_factor);
  imresize(img, new_size);
  return vector<float32_t >{scale_factor};
}

template<typename Dtype>
void imnormalize(cv::Mat& img, vector<Dtype> mean, vector<Dtype> std, bool_t to_rgb){
  img.convertTo(img, CV_32FC3);
  if(to_rgb)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  CHECK(img.channels() == 3);
  for (int i = 0; i < img.rows; ++i)
    for (int j = 0; j < img.cols; ++j)
      for (int n = 0; n < 3; ++n) {
        img.at<cv::Vec3f>(i, j)[n] = (img.at<cv::Vec3f>(i, j)[n] - mean[n]) / std[n];
      }
}

inline void imflip(cv::Mat & img, int direction = 1){
  cv::flip(img, img, direction);
}

cv::Mat impad(cv::Mat img, int32_t pad_h, int32_t pad_w, int32_t pad_val){
  CHECK(pad_h >= img.rows && pad_w >= img.cols && img.channels() == 3);

  cv::Mat new_img = cv::Mat::zeros(pad_h, pad_w, CV_32FC3);

  for (int i = 0; i < img.rows; ++i)
    for (int j = 0; j < img.cols; ++j)
      for (int n = 0; n < 3; ++n){
        new_img.at<cv::Vec3f>(i, j)[n] = img.at<cv::Vec3f>(i, j)[n];
      }
  return new_img;
}

cv::Mat impad_to_multiple(cv::Mat img, int32_t size_divisor, int32_t pad_val = 0){
  int32_t pad_h = int(std::ceil(float32_t(img.rows) / size_divisor)) * size_divisor;
  int32_t pad_w = int(std::ceil(float32_t(img.cols) / size_divisor)) * size_divisor;
  return impad(img, pad_h, pad_w, pad_val);
}

ImgInfo<float32_t> img2float(cv::Mat img){
  float32_t * img_float = new float32_t[3 * img.rows * img.cols];
  for (int n = 0; n < 3; ++n)
    for (int i = 0; i < img.rows; ++i)
      for (int j = 0; j < img.cols; ++j)
        img_float[n * img.rows * img.cols + i * img.cols + j] = img.at<cv::Vec3f>(i, j)[n];
  ImgInfo<float32_t> img_info;
  img_info.img = img_float;
  img_info.pad_shape = { img.rows, img.cols };

  return img_info;
}

template <typename Dtype >
ImagePreprocess<Dtype>::ImagePreprocess(
    const ImgPreProcessStruct<Dtype>& img_preprocess_struct) {
  img_preprocess_struct_ = img_preprocess_struct;
}

template <typename Dtype >
ImgInfo<Dtype> ImagePreprocess<Dtype>::ImageTransform(const std::string& path) {
  ImgInfo<Dtype> img_info;
  cv::Mat new_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  CHECK(!new_img.empty());
  vector<Dtype> scale_factor;
  vector<int32_t> scale(2);
  scale[0] = this->img_preprocess_struct_.scale.h;
  scale[1] = this->img_preprocess_struct_.scale.w;
  if (this->img_preprocess_struct_.resize_keep_ratio) {
    vector<float32_t> temp = imrescale(new_img, scale);
    for (int32_t i = 0; i < temp.size(); ++i)
      scale_factor.push_back(temp[i]);
  } else {
    vector<float32_t > scale_ = imresize(new_img, scale);
    for(int32_t i = 0; i < 4; ++i)
      scale_factor.push_back(scale_[i % 2]);
  }
  img_info.img_shape = vector<int32_t>{ new_img.rows, new_img.cols };

  imnormalize(new_img, this->img_preprocess_struct_.image_mean,
      this->img_preprocess_struct_.image_std, this->img_preprocess_struct_.to_rgb);

  if (this->img_preprocess_struct_.flip)
    imflip(new_img);

  if(this->img_preprocess_struct_.size_divisor > 0)
    img_info = img2float(impad_to_multiple(new_img, this->img_preprocess_struct_.size_divisor));
  else
    img_info = img2float(new_img);

  img_info.img_shape = vector<int32_t>{ new_img.rows, new_img.cols };
  img_info.scale_factor = scale_factor;

  return img_info;
}

template class ImagePreprocess<float32_t>;

}  // namespace cobjectflow

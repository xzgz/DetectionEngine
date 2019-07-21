#ifndef USE_OPENCV
#define USE_OPENCV
#endif
//#ifndef CPU_ONLY
//#define CPU_ONLY
//#endif

#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include<complex>
#include<cstdlib>

#include "cnpy.h"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using boost::shared_ptr;
using std::cout;
using std::endl;
using std::setw;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  Blob<float> *Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void PrintBlobs(vector<Blob<float>*>& blobs);
  void PrintOneBlob(Blob<float> *blob);
  void ResetOneBlob(Blob<float> *blob);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  string trained_file_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
//  net_.reset(new Net<float>(model_file, TRAIN));
  trained_file_ = trained_file;
//  net_->CopyTrainedLayersFrom(trained_file);

//  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
//  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
  << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
//  SetMean(mean_file);

  /* Load labels. */
//  std::ifstream labels(label_file.c_str());
//  CHECK(labels) << "Unable to open labels file " << label_file;
//  string line;
//  while (std::getline(labels, line))
//    labels_.push_back(string(line));

//  Blob<float>* output_layer = net_->output_blobs()[0];
//  CHECK_EQ(labels_.size(), output_layer->channels())
//    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  Blob<float> *output = Predict(img);

  N = std::min<int>(labels_.size(), N);
//  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
//  for (int i = 0; i < N; ++i) {
//    int idx = maxN[i];
//    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
//  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

Blob<float> *Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  cnpy::NpyArray arr = cnpy::npy_load("/home/SENSETIME/heyanguang/code/mask-rcnn/caffe-output/source.npy");
  float *loaded_data = arr.data<float>();
  float *input_data = input_layer->mutable_cpu_data();

  for (int i = 0; i < input_layer->count(0); ++i) {
    input_data[i] = loaded_data[i];
  }

//  std::vector<cv::Mat> input_channels;
//  WrapInputLayer(&input_channels);
//  Preprocess(img, &input_channels);

  net_->CopyTrainedLayersFrom(trained_file_);
//  PrintOneBlob(input_layer);

  // DEPRECATED: Forward(bottom, loss) will be removed in a future version.
  // Use Forward(loss).
  // net_->Forward(net_->input_blobs());
  net_->Forward(net_->input_blobs());

  vector<float> params_lr = net_->params_lr();
//  vector<Blob<float>*> learnable_params = net_->learnable_params();
  vector<shared_ptr<Blob<float> > > params = net_->params();

  shared_ptr<Layer<float> > Convolution_0 = net_->layer_by_name("Convolution_0");
  vector<shared_ptr<Blob<float> > > Convolution_0_blobs = Convolution_0->blobs();
//  ResetOneBlob(Convolution_0_blobs[0].get());
//  ResetOneBlob(Convolution_0_blobs[1].get());

//  net_->CopyTrainedLayersFrom(trained_file_);

//  PrintOneBlob(Convolution_0_blobs[0].get());
//  PrintOneBlob(Convolution_0_blobs[1].get());

  vector<string> blob_names = net_->blob_names();
  vector<vector<Blob<float>*> > bottom_vecs = net_->bottom_vecs();
  vector<vector<Blob<float>*> > top_vecs = net_->top_vecs();

  shared_ptr<Blob<float> > input_0 = net_->blob_by_name("0");
//  shared_ptr<Blob<float> > conv0_346 = net_->blob_by_name("346");
//  shared_ptr<Blob<float> > interp_525 = net_->blob_by_name("525");
//  shared_ptr<Blob<float> > eltwise_526 = net_->blob_by_name("526");
//  shared_ptr<Blob<float> > eltwise_524 = net_->blob_by_name("524");
//  shared_ptr<Blob<float> > eltwise_522 = net_->blob_by_name("522");
//  shared_ptr<Blob<float> > conv_517 = net_->blob_by_name("517");
//  shared_ptr<Blob<float> > conv_518 = net_->blob_by_name("518");
//  shared_ptr<Blob<float> > conv_519 = net_->blob_by_name("519");
//  shared_ptr<Blob<float> > conv_520 = net_->blob_by_name("520");
//  shared_ptr<Blob<float> > out_536 = net_->blob_by_name("536");

  shared_ptr<Blob<float> > target_blob = net_->blob_by_name("521");

//  shared_ptr<Blob<float> > target_blob = blob_482;
  PrintOneBlob(target_blob.get());
//  std::ofstream fid("/home/SENSETIME/heyanguang/code/mask-rcnn/caffe-output/data_blob.txt", std::ios::out);
//  if (target_blob->num_axes() == 1)
//  {
//    for (int n = 0; n < target_blob->count(); ++n) {
//      fid << target_blob.get()->cpu_data()[n] << " ";
//    }
//  } else {
//    int num = target_blob->num();
//    int height = target_blob->height();
//    int width = target_blob->width();
//    int channel = target_blob->channels();
//    fid << channel << " " << width << " " << height << "\n";
//    for (int n = 0; n < num; ++n) {
//      for (int c = 0; c < channel; ++c) {
//        for (int h = 0; h < height; ++h) {
//          for (int w = 0; w < width; ++w)
//            fid << target_blob->data_at(n, c, h, w) << " ";
//        }
//        fid << "\n";
//      }
//      fid << "\n";
//    }
//  }
//  fid.close();



//  PrintOneBlob(conv0_346.get());
//  PrintOneBlob(input_0.get());
//  PrintOneBlob(eltwise_522.get());
//  PrintOneBlob(eltwise_526.get());
//  PrintOneBlob(interp_525.get());
//  PrintOneBlob(conv_517.get());
//  PrintOneBlob(conv_518.get());
//  PrintOneBlob(conv_519.get());
//  PrintOneBlob(conv_520.get());
//  PrintOneBlob(out_536.get());

  /* Copy the output layer to a std::vector */
  vector<Blob<float>*> output_layer = net_->output_blobs();
//  PrintBlobs(output_layer);
//  PrintOneBlob(output_layer[0]);
//  const float* begin = output_layer->cpu_data();
//  const float* end = begin + output_layer->channels();
//  return std::vector<float>(begin, end);
  return output_layer[0];
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

//  cv::Mat sample_normalized;
//  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
  << "Input channels are not wrapping the input layer of the network.";
}

void Classifier::PrintBlobs(vector<Blob<float>*>& blobs) {
  for (int i = 0; i < blobs.size(); ++i) {
    cout << "blob: " << i << endl;
    int N = std::min(10, blobs[i]->shape(0));
    int C = std::min(3, blobs[i]->shape(1));
    int H = std::min(10, blobs[i]->shape(2));
    int W = std::min(10, blobs[i]->shape(3));
    const float *data = blobs[i]->cpu_data();
    int C2W = blobs[i]->count(1);
    int H2W = blobs[i]->count(2);
    for (int n = 0; n < N; ++n) {
      cout << "n: " << n << endl;
      for (int c = 0; c < C; ++c) {
        cout << "c: " << c << endl;
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            cout << setw(12) << data[n * C2W + c * H2W + h * W + w];
          }
          cout << "\t";
        }
        cout << endl;
      }
    }
  }
}

void Classifier::PrintOneBlob(Blob<float> *blob) {
  const float *data = blob->cpu_data();
  if (blob->num_axes() == 1) {
    cout << "blob shape: " << blob->shape(0) << endl;
    int nw = std::min(10, blob->shape(0));
    for (int i = 0; i < nw; ++i) {
      cout << setw(12) << data[i];
    }
    cout << endl;
    return;
  }

  cout << "blob shape: " << blob->shape(0) << " "
                         << blob->shape(1) << " "
                         << blob->shape(2) << " "
                         << blob->shape(3) << endl;
  int nw = std::min(10, blob->shape(0));
  int cw = std::min(3, blob->shape(1));
  int hw = std::min(10, blob->shape(2));
  int ww = std::min(10, blob->shape(3));
  int c2w_cnt = blob->count(1);
  int h2w_cnt = blob->count(2);
  int w_cnt = blob->count(3);
  for (int n = 0; n < nw; ++n) {
    cout << "n: " << n << endl;
    for (int c = 0; c < cw; ++c) {
      cout << "c: " << c << endl;
      for (int h = 0; h < hw; ++h) {
        for (int w = 0; w < ww; ++w) {
          cout << setw(12) << data[n * c2w_cnt + c * h2w_cnt + h * w_cnt + w];
        }
        cout << "\t";
      }
      cout << endl;
    }
  }
}

void Classifier::ResetOneBlob(Blob<float> *blob) {
  float *data = blob->mutable_cpu_data();
  if (blob->num_axes() == 1) {
    int nw = std::min(10, blob->shape(0));
    for (int i = 0; i < nw; ++i) {
      data[i] = 2;
    }
    return;
  }

  int nw = std::min(10, blob->shape(0));
  int cw = std::min(3, blob->shape(1));
  int hw = std::min(10, blob->shape(2));
  int ww = std::min(10, blob->shape(3));
  int c2w_cnt = blob->count(1);
  int h2w_cnt = blob->count(2);
  int w_cnt = blob->count(3);
  for (int n = 0; n < nw; ++n) {
    for (int c = 0; c < cw; ++c) {
      for (int h = 0; h < hw; ++h) {
        for (int w = 0; w < ww; ++w) {
          data[n * c2w_cnt + c * h2w_cnt + h * w_cnt + w] = 2;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
//  if (argc != 6) {
//    std::cerr << "Usage: " << argv[0]
//              << " deploy.prototxt network.caffemodel"
//              << " mean.binaryproto labels.txt img.jpg" << std::endl;
//    return 1;
//  }

//  ::google::InitGoogleLogging(argv[0]);

  string model_file   = "/home/SENSETIME/heyanguang/code/mask-rcnn/caffe-output/rel.prototxt";
  string trained_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/caffe-output/model.bin";
//  string model_file   = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/retinanet_ratio32_bilinear/retinanet_32_bilinear.prototxt";
//  string trained_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/retinanet_ratio32_bilinear/retinanet_32_bilinear.caffemodel";
//  string trained_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/retinanet_ratio32_bilinear/retinanet_32_bilinear_2.caffemodel";
//  string trained_file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/retinanet_ratio32_bilinear/converted_model_stage1.caffemodel";
  string mean_file    = "";
  string label_file   = "";
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = "/home/SENSETIME/heyanguang/code/mask-rcnn/CObjectFlow/retinanet_ratio32_bilinear/finger.jpg";

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  cv::cvtColor(img, img, CV_BGR2RGB);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

//  /* Print the top N predictions. */
//  for (size_t i = 0; i < predictions.size(); ++i) {
//    Prediction p = predictions[i];
//    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//              << p.first << "\"" << std::endl;
//  }
}

#endif  // USE_OPENCV

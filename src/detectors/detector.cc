#include <iomanip>

#include "detectors/detector.h"

using std::cout;
using std::endl;
using std::setw;

namespace cobjectflow {

template <typename Dtype>
Detector<Dtype>::Detector(
    const std::string& proto_txt, const std::string& model_param,
    const Phase phase) {
  std::string proto_txt_file = proto_txt;
  std::string model_param_file = model_param;
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  net_.reset(new Net<Dtype>(proto_txt_file, phase));
  net_->Reshape();
  net_->CopyTrainedLayersFrom(model_param_file);
}

template <typename Dtype>
void Detector<Dtype>::NetForward() {
  net_->Forward();
}

template <typename Dtype>
const std::vector<Blob<Dtype>*>& Detector<Dtype>::output_feature_blobs() {
  return net_->output_blobs();
}

template <typename Dtype>
Blob<Dtype> *Detector<Dtype>::blob_by_name(const std::string& blob_name) {
  return net_->blob_by_name(blob_name).get();
}

template <typename Dtype>
const std::vector<Blob<Dtype>*> Detector<Dtype>::blobs_by_names(
    const std::vector<std::string>& output_blob_names) {
  CHECK(output_blob_names.size() > 0);
  std::vector<Blob<Dtype>*> by_name_blobs(output_blob_names.size());
  for (uint32_t i = 0U; i < output_blob_names.size(); ++i) {
    by_name_blobs[i] = net_->blob_by_name(output_blob_names[i]).get();
  }

  return std::move(by_name_blobs);
}

template <typename Dtype>
void Detector<Dtype>::SetInputBlobs(
    const std::vector<Dtype *>& input_datas,
    const std::vector<int32_t>& input_data_counts) {
  auto input_blobs = net_->input_blobs();
  auto input_blob_num = input_blobs.size();
  CHECK(input_blob_num == input_datas.size() &&
        input_blob_num == input_data_counts.size());

  for (uint32_t i = 0; i < input_blob_num; ++i) {
    CHECK(input_blobs[i]->count(0) == input_data_counts[i]);
    input_blobs[i]->set_cpu_data(input_datas[i]);
  }
}

template <typename Dtype>
void Detector<Dtype>::ReshapeNet(const std::vector<int32_t>& shape) {
  auto input_blobs = net_->input_blobs();
  // only consider input blobs number is 1.
  input_blobs[0]->Reshape(shape);
  net_->Reshape();
}

template <typename Dtype>
void Detector<Dtype>::PrintOneBlob(const std::string& blob_name) {
  boost::shared_ptr<Blob<Dtype> > blob = net_->blob_by_name(blob_name);
  PrintOneBlob(blob.get());
}

template <typename Dtype>
void Detector<Dtype>::PrintOneBlob(const Blob<Dtype> *blob) {
  CHECK(blob->num_axes() == 1 || blob->num_axes() == 2 || blob->num_axes() == 4);

  const float *data = blob->cpu_data();
  if (blob->num_axes() == 1) {
    cout << "blob shape: ";
    for (int an = 0; an < blob->shape().size(); ++an) {
      cout << blob->shape(an) << " ";
    }
    cout << endl;
    int nw = std::min(10, blob->shape(0));
    for (int i = 0; i < nw; ++i) {
      cout << setw(12) << data[i];
    }
    cout << endl;
  } else if (blob->num_axes() == 2) {
    cout << "blob shape: ";
    for (int an = 0; an < blob->shape().size(); ++an) {
      cout << blob->shape(an) << " ";
    }
    cout << endl;
    int nw = std::min(10, blob->shape(0));
    int cw = std::min(10, blob->shape(1));
    for (int n = 0; n < nw; ++n) {
      cout << "n: " << n << endl;
      for (int c = 0; c < cw; ++c) {
        cout << setw(12) << data[n * cw + c];
      }
      cout << endl;
    }
  } else {
    cout << "blob shape: ";
    for (int an = 0; an < blob->shape().size(); ++an) {
      cout << blob->shape(an) << " ";
    }
    cout << endl;
    int nw = std::min(100, blob->shape(0));
    int cw = std::min(3, blob->shape(1));
    int hw = std::min(10, blob->shape(2));
    int ww = std::min(10, blob->shape(3));
    int c2w_cnt = blob->count(1);
    int h2w_cnt = blob->count(2);
    int w_cnt = blob->count(3);
    for (int n = 95; n < nw; ++n) {
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
}

template class Detector<float32_t>;

}  // namespace cobjectflow

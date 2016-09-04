// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  sigma2_ = loss_param.sigma() * loss_param.sigma();
  has_weights_ = (bottom.size() >= 3);
  if (has_weights_) {
    CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

// LIOR NOTES: switched bottom[0] to bottom[1] to enable hackey
// training trick where i stack new cls/reg layers over old ones
template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[1]->height(), bottom[1]->height());
  CHECK_EQ(bottom[1]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[1]->height(), bottom[2]->height());
    CHECK_EQ(bottom[1]->width(), bottom[2]->width());
    CHECK_EQ(bottom[1]->channels(), bottom[3]->channels());
    CHECK_EQ(bottom[1]->height(), bottom[3]->height());
    CHECK_EQ(bottom[1]->width(), bottom[3]->width());
  }
  diff_.Reshape(bottom[1]->num(), bottom[1]->channels(),
      bottom[1]->height(), bottom[1]->width());
  errors_.Reshape(bottom[1]->num(), bottom[1]->channels(),
      bottom[1]->height(), bottom[1]->width());
  // vector of ones used to sum
  ones_.Reshape(bottom[1]->num(), bottom[1]->channels(),
      bottom[1]->height(), bottom[1]->width());
  for (int i = 0; i < bottom[1]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1LossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1LossLayer);
REGISTER_LAYER_CLASS(SmoothL1Loss);

}  // namespace caffe

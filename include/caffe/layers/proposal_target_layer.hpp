#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/proposal_common.hpp"

#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
class ProposalTargetLayer : public Layer<Dtype> {

public:
explicit ProposalTargetLayer(LayerParameter const & param): Layer<Dtype>(param), iteration_(0) {}

virtual void LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                        std::vector<Blob<Dtype>*> const & top);

virtual void Reshape(std::vector<Blob<Dtype>*> const & bottom,
                     std::vector<Blob<Dtype>*> const & top);

virtual inline const char * type() const { return "ProposalTargetLayer"; }

protected:
virtual void Forward_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                         std::vector<Blob<Dtype>*>  const & top);

virtual void Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                          std::vector<bool>         const & propagate_down,
                          std::vector<Blob<Dtype>*> const & bottom);

protected:
  cv::Size layer_size_;
  cv::Size image_size_;

  std::vector<cv::Rect_<Dtype>> anchors_;

  ProposalTargetParameter params_;
  int iteration_;

  std::vector<size_t> keep_indices_;
};

}

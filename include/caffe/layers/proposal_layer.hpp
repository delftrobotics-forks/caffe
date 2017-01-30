#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/proposal_common.hpp"

#include <cstdlib>
#include <initializer_list>
#include <iomanip>

namespace caffe {

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {

public:
explicit ProposalLayer(LayerParameter const & param): Layer<Dtype>(param) {}

virtual void LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                        std::vector<Blob<Dtype>*> const & top);

virtual void Reshape(std::vector<Blob<Dtype>*> const & bottom,
                     std::vector<Blob<Dtype>*> const & top);

virtual inline const char * type() const { return "ProposalLayer"; }


protected:
virtual void Forward_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                         std::vector<Blob<Dtype>*>  const & top);

virtual void Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                          std::vector<bool>         const & propagate_down,
                          std::vector<Blob<Dtype>*> const & bottom);

private:
std::vector<Dtype> generateScoresVector(Blob<Dtype> const & blob, int const offset);

protected:
  cv::Size layer_size_;
  cv::Size image_size_;

  std::vector<cv::Rect_<Dtype>> anchors_;

  ProposalParameter parameters_;

  std::vector<size_t> anchor_index_before_clip_;
  std::vector<size_t> proposal_index_;
  std::vector<size_t> proposal_index_before_clip_;
  std::vector<size_t> ind_after_filter_;
  std::vector<size_t> ind_after_sort_;
};

}

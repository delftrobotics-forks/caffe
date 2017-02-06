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
class MaskProposalLayer : public Layer<Dtype> {

public:
explicit MaskProposalLayer(LayerParameter const & param): Layer<Dtype>(param) {}

virtual void LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                        std::vector<Blob<Dtype>*> const & top);

virtual void Reshape(std::vector<Blob<Dtype>*> const & bottom,
                     std::vector<Blob<Dtype>*> const & top);

virtual inline const char * type() const { return "MaskProposalLayer"; }


protected:
virtual void Forward_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                         std::vector<Blob<Dtype>*>  const & top);

void ForwardTrain_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                      std::vector<Blob<Dtype>*>  const & top,
                      std::vector<cv::Mat>             & resized_mask_pred);

void ForwardTest_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                     std::vector<Blob<Dtype>*>  const & top,
                     cv::Mat & resized_mask_pred);

virtual void Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                          std::vector<bool>         const & propagate_down,
                          std::vector<Blob<Dtype>*> const & bottom);

MaskProposalParameter parameters_;

cv::Mat pos_sample_;

};

}


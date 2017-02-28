#pragma once

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include <opencv2/opencv.hpp>


namespace caffe {

/**
 * Assign anchors to ground-truth targets. Produces anchor classification
 * labels and bounding-box regression targets.
*/
template <typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {
  public:
  explicit AnchorTargetLayer(LayerParameter const & param): Layer<Dtype>(param) {}

  virtual void LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                          std::vector<Blob<Dtype>*> const & top);

  virtual void Reshape(std::vector<Blob<Dtype>*> const & bottom,
                       std::vector<Blob<Dtype>*> const & top);

  virtual inline const char * type() const { return "AnchorTargetLayer"; }

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

    AnchorTargetParameter parameters_;
};

}

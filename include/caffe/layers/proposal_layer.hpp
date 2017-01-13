#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>
#include <numeric>

namespace caffe {

template <typename T>
class Rectangle : public cv::Rect_<T> {
public:

  using cv::Rect_<T>::Rect_;

  Rectangle & operator&(Rectangle const & other) {
    T tl_x = std::max(this->tl().x, other.tl().x);
    T tl_y = std::max(this->tl().y, other.tl().y);
    T br_x = std::min(this->br().x, other.br().x);
    T br_y = std::min(this->br().y, other.br().y);

    this->x      = tl_x;
    this->y      = tl_y;
    this->width  = br_x - tl_x;
    this->height = br_y - tl_y;

    return *this;
  }

  cv::Point_<T> center() const { return { this->x + this->width / 2, this->y + this->height / 2 }; }

  static Rectangle<T> centered(T const x, T const y, T const width, T const height) {
    return { x - width / 2, y - height / 2, width, height };
  }

  static Rectangle<T> centered(cv::Point_<T> const center, T const width, T const height) {
    return Rectangle<T>::centered(center.x, center.y, width, height);
  }

  static Rectangle<T> centered(cv::Point_<T> const center, cv::Size_<T> const size) {
    return Rectangle<T>::centered(center.x, center.y, size.width, size.height);
  }

  friend std::ostream & operator<<(std::ostream & stream, Rectangle const & r) {
    stream << "[" << r.tl().x << ", " << r.tl().y << ", " << r.br().x << ", " << r.br().y << "]";
    return stream;
  }
};

typedef Rectangle<float>  Rectanglef;
typedef Rectangle<double> Rectangled;
typedef Rectangle<int>    Rectanglei;
 
namespace proposal_layer {
  template <typename T>
  void print(std::vector<T> const & vector) {
    for (auto const & element : vector) { std::cout << element << " "; }
    std::cout << std::endl;
  }

  template <typename T>
  std::vector<T> select(std::vector<T> const & vector, std::vector<size_t> const & indices) {
    std::vector<T> result;
    std::transform(indices.begin(), indices.end(), std::back_inserter(result), [vector](size_t i) { return vector[i]; });
    return result;
  }
}

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
  template <typename T>
  std::vector<size_t> stableSort(std::vector<T> const & v) const {
    std::vector<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&v](size_t i, size_t j) { return v[i] >= v[j]; });
    return indices;
  }

  std::vector<Rectanglef> generateBaseAnchors(std::vector<float> const & ratios,
                                              std::vector<float> const & scales,
                                              int                const   base_size);

  std::vector<Rectanglef> generateShiftedAnchors(std::vector<Rectanglef> const & base_anchors,
                                                 cv::Size                const   layer_size,
                                                 int                     const   feat_stride);

  std::vector<size_t> clipRectangles(std::vector<Rectanglef>       & rectangles,
                                     cv::Size                const   image_size,
                                     bool                    const   auto_clip = false) const;

  std::vector<size_t> getLargeRectangles(std::vector<Rectanglef> const & rectangles,
                                         float                   const   min_size) const;

  std::vector<size_t> applyNonMaximumSuppression(std::vector<Rectanglef> const & proposals,
                                                 std::vector<Dtype>      const & scores,
                                                 float                   const   threshold) const;

  std::vector<Rectanglef> generateProposals(std::vector<Rectanglef> const & anchors,
                                            Blob<Dtype>             const * deltas);

  Rectanglef generateAnchorByRatio(Rectanglef const & anchor, float const ratio = 1.0) const;
  Rectanglef generateAnchorByScale(Rectanglef const & anchor, float const scale = 1.0) const;
  std::vector<Dtype> generateScoresVector(Blob<Dtype> const & blob, int const offset) const;

  protected:
    cv::Size layer_size_;
    cv::Size image_size_;

    std::vector<Rectanglef> anchors_;

    ProposalParameter parameters_;

    std::vector<size_t> anchor_index_before_clip_;
    std::vector<size_t> proposal_index_;
    std::vector<size_t> proposal_index_before_clip_;
    std::vector<size_t> ind_after_filter_;
    std::vector<size_t> ind_after_sort_;
};

}

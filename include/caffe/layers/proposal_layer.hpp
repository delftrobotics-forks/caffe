#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <iomanip>

namespace caffe {

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

  struct Point {
    float x;
    float y;

    Point(float _x, float _y)  { x = _x;      y = _y;      }
    Point(Point const & other) { x = other.x; y = other.y; }

    friend std::ostream & operator<<(std::ostream & stream, Point const & point) {
      stream << "[ x: " << std::setw(8) << point.x << ", y: " << std::setw(8) << point.y << "]";
      return stream;
    }
  };

  struct Rectangle {
    Point center;
    float width;
    float height;

    Rectangle(float _x, float _y, float _width, float _height): center(_x, _y),  width(_width), height(_height) {}
    Rectangle(Point _center, float _width, float _height):      center(_center), width(_width), height(_height) {}

    Rectangle intersect(Rectangle const & rect) const {
      Point top_left     = this->topLeft();
      Point bottom_right = this->bottomRight();

      float x1 = std::max(top_left.x,     rect.topLeft().x);
      float y1 = std::max(top_left.y,     rect.topLeft().y);
      float x2 = std::min(bottom_right.x, rect.bottomRight().x);
      float y2 = std::min(bottom_right.y, rect.bottomRight().y);

      if (x1 > x2 || y1 > y2) { return Rectangle(0, 0, 0, 0); }

      return Rectangle::fromCoordinates(x1, y1, x2, y2);
    }

    friend std::ostream & operator<<(std::ostream & stream, Rectangle const & anchor) {
      //stream << anchor.center << " [ w: " << std::setw(6) << anchor.width  <<
                                 //", h: "  << std::setw(6) << anchor.height << " ]";
      stream << anchor.topLeft() << " " << anchor.bottomRight();
      return stream;
    }

    static Rectangle fromCoordinates(float px, float py, float rx, float ry) {
      return Rectangle::fromCoordinates(Point(px, py), Point(rx, ry));
    }

    static Rectangle fromCoordinates(Point top_left, Point bottom_right) {
      float width  = bottom_right.x - top_left.x + 1;
      float height = bottom_right.y - top_left.y + 1;

      Point center(top_left.x + 0.5 * (width  - 1), top_left.y + 0.5 * (height - 1));

      return Rectangle(center, width, height);
    }

    float area() const { return width * height; }

    Point topLeft() const {
      return Point(center.x - 0.5 * (width  - 1), center.y - 0.5 * (height - 1));
    }

    Point bottomRight() const {
      return Point(center.x + 0.5 * (width  - 1), center.y + 0.5 * (height - 1));
    }
  };
}

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {

  using Rectangle = proposal_layer::Rectangle;
  using Point  = proposal_layer::Point;

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

  void generateReferenceAnchors(std::vector<Rectangle>       & anchors,
                                std::vector<float>     const & ratios,
                                std::vector<float>     const & scales,
                                int                    const   base_size);

  void generateShiftedAnchors(std::vector<Rectangle>       & shifted_anchors,
                              std::vector<Rectangle> const & reference_anchors,
                              int                    const   layer_width,
                              int                    const   layer_height,
                              int                    const   feat_stride);

  void clipRectanglesToBounds(std::vector<size_t>          & indices,
                              std::vector<Rectangle>       & rectangles,
                              int                    const   width,
                              int                    const   height,
                              bool                   const   auto_clip = false);

  std::vector<size_t> getLargeRectangles(std::vector<Rectangle> const & rectangles,
                                         float                  const   min_size);

  std::vector<size_t> applyNonMaximumSuppression(std::vector<Rectangle> const & proposals,
                                                 std::vector<float>     const & scores,
                                                 float                  const   threshold,
                                                 size_t                 const   pre_nms_top_n  = 6000,
                                                 size_t                 const   post_nms_top_n = 300) const;

  bool generateProposals(std::vector<Rectangle>          & proposals,
                         std::vector<Rectangle>    const & anchors,
                         Blob<Dtype>               const * deltas);

  Rectangle generateAnchorByRatio(Rectangle const & anchor, float const ratio = 1.0);
  Rectangle generateAnchorByScale(Rectangle const & anchor, float const scale = 1.0);
  std::vector<float> generateScoresVector(Blob<Dtype> const & blob, int const offset) const;

  protected:
    int feat_stride_;
    float clip_denominator_;
    float clip_threshold_;
    bool use_clip_;

    std::vector<Rectangle> reference_anchors_;
    std::vector<size_t> anchor_indices_;
    std::vector<size_t> proposal_indices_;
    std::vector<size_t> filter_indices_;
};

}

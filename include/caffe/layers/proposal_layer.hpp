#pragma once

#include <initializer_list>
#include <iomanip>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

namespace proposal_layer {

  template <typename T>
    void printVector(std::vector<T> const & vector) {
      for (auto const & element : vector) { std::cout << element << " "; }
      std::cout << std::endl;
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

    Rectangle(float _x, float _y, float _width, float _height):
      center(_x, _y),  width(_width), height(_height) {}
    Rectangle(Point _center, float _width, float _height):
      center(_center), width(_width), height(_height) {}

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
  virtual void Reshape(std::vector<Blob<Dtype>*>    const & bottom,
                       std::vector<Blob<Dtype>*>    const & top);

  virtual inline const char * type() const { return "ProposalLayer"; }

  protected:
  virtual void Forward_cpu(std::vector<Blob<Dtype>*>  const & bottom,
                           std::vector<Blob<Dtype>*>  const & top);
  virtual void Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                            std::vector<bool>         const & propagate_down,
                            std::vector<Blob<Dtype>*> const & bottom);

  private:
  void generateReferenceAnchors(std::vector<Rectangle>       & anchors,
                                std::vector<float>     const & ratios,
                                std::vector<float>     const & scales,
                                int                    const   base_size);

  void generateShiftedAnchors(std::vector<Rectangle>       & shifted_anchors,
                              std::vector<Rectangle> const & reference_anchors,
                              int                    const   layer_width,
                              int                    const   layer_height,
                              int                    const   feat_stride);

  void clipRectanglesToBounds(std::vector<int>             & indexes,
                              std::vector<Rectangle>       & rectangles,
                              int                    const   width,
                              int                    const   height,
                              bool                   const   auto_clip = false);

  std::vector<int> getLargeRectangles(std::vector<Rectangle> const & rectangles,
                                      float                  const   min_size);

  bool generateProposals(std::vector<Rectangle>          & proposals,
                         std::vector<Rectangle>    const & anchors,
                         Blob<Dtype>               const * deltas);

  Rectangle generateAnchorByRatio(Rectangle const & anchor, float const ratio = 1.0);
  Rectangle generateAnchorByScale(Rectangle const & anchor, float const scale = 1.0);
  std::vector<float> generateScoresVector(Blob<Dtype> const & blob, int const offset);

  protected:
    int feat_stride_;
    float clip_denominator_;
    float clip_threshold_;
    bool use_clip_;

    std::vector<Rectangle> reference_anchors_;
    std::vector<int> anchor_indexes_;
    std::vector<int> proposal_indexes_;
    std::vector<int> filter_indexes_;
    
};

}

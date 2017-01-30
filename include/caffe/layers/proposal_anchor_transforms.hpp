#pragma once

#include "proposal_common.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {
namespace proposal_layer {
namespace anchor {
  /** \brief Generates an anchor with the same area as an reference one and a given aspect ratio.
   *  \param [in]  anchor   Anchor that is used as reference.
   *  \param [in]  ratio    Aspect ratio of the output anchor.
   *  \return Anchor with the given aspect ratio and same (reference) area.
   */
  template <typename T>
  cv::Rect_<T> generateAnchorByRatio(cv::Rect_<T> const & anchor, float const ratio = 1.0) {
    cv::Point_<T> center = rectangle::getRectangleCenter(anchor);
    T width              = std::round(std::sqrt(anchor.area() / ratio));
    cv::Size_<T> size    = cv::Size_<T>(width, std::round(width * ratio));

    return rectangle::centeredRectangle(center.x, center.y, size.width, size.height);
  }

  /** \brief Scales up an reference anchor, by a given factor.
   *  \param [in]  anchor   Anchor that is used as reference (to be scaled up).
   *  \param [in]  scale    Factor by which the reference anchor is scaled up.
   *  \return Scaled up anchor.
   */
  template <typename T>
  cv::Rect_<T> generateAnchorByScale(cv::Rect_<T> const & anchor, float const scale = 1.0) {
    cv::Point_<T> center = rectangle::getRectangleCenter(anchor);
    return rectangle::centeredRectangle(center.x, center.y, anchor.width * scale, anchor.height * scale);
  }

  /** \brief Generates anchors with different ratios and scales, starting from a
   *         square anchor with size 'base_size', centered at (0, 0).
   *  \param [in]   ratios      Vector with desired anchor ratios.
   *  \param [in]   scales      Vector with desired anchor scales.
   *  \param [in]   base_size   Size of the (square) base anchor.
   *  \return Vector with the generated anchors.
   */
  template <typename T>
  std::vector<cv::Rect_<T>> generateBaseAnchors(std::vector<float> const & ratios    = { 0.5, 1, 2 },
                                                std::vector<float> const & scales    = { 8, 16, 32 },
                                                int                const   base_size = 16)
  {
    cv::Rect_<T> base_anchor(0, 0, base_size, base_size);

    std::vector<cv::Rect_<T>> anchors;
    anchors.reserve(ratios.size() * scales.size());

    for (auto const ratio : ratios) {
      cv::Rect_<T> ratio_anchor = generateAnchorByRatio<T>(base_anchor, ratio);
      for (auto const scale : scales) {
        anchors.push_back(generateAnchorByScale(ratio_anchor, scale));
      }
    }

    return anchors;
  }

  /** \brief Shifts a set of reference anchors, such that they "cover" the whole surface
   *         of a WxH grid, where W and H are the width and height of the bottom layer.
   *  \param [in]  reference_anchors   Vector with the reference (input) anchors.
   *  \param [in]  layer_size          Size of the bottom layer.
   *  \param [in]  feat_stride         Stride used when generating the shifted anchors.
   *  \return Vector with the generated (shifted) anchors.
   */
  template <typename T>
  std::vector<cv::Rect_<T>> generateShiftedAnchors(std::vector<cv::Rect_<T>> const & base_anchors,
                                                   cv::Size                  const   layer_size,
                                                   int                       const   feat_stride)
  {
    std::vector<cv::Rect_<T>> anchors;
    anchors.reserve(layer_size.width * layer_size.height * base_anchors.size());

    for (int y = 0; y < layer_size.height; ++y) {
      for (int x = 0; x < layer_size.width; ++x) {
        for (auto const & anchor : base_anchors) {
          cv::Point_<T> tl = anchor.tl() + cv::Point_<T>(x * feat_stride, y * feat_stride);
          cv::Point_<T> br = anchor.br() + cv::Point_<T>(x * feat_stride, y * feat_stride);
          anchors.push_back({ tl, br });
        }
      }
    }

    return anchors;
  }

  /** \brief Applies delta transformation on given anchors, to get transformed proposals.
   *  \param [in]  anchors   Vector of anchors.
   *  \param [in]  deltas    Blob with delta transformations.
   *  \return Vector of proposals.
   */
  template <typename T>
  std::vector<cv::Rect_<T>> generateProposals(std::vector<cv::Rect_<T>> const & anchors,
                                              Blob<T>                   const * deltas)
  {
    std::vector<int> shape = deltas->shape();
    std::vector<cv::Rect_<T>> proposals;

    int index = 0;

    proposals.reserve(shape[1] * shape[2] * shape[3] / 4);
    for (int x = 0; x < deltas->shape()[2]; ++x) {
      for (int y = 0; y < deltas->shape()[3]; ++y) {
        for (int b = 0; b < deltas->shape()[1]; b += 4) {
          T dx = deltas->data_at(0, b,     x, y);
          T dy = deltas->data_at(0, b + 1, x, y);
          T dw = deltas->data_at(0, b + 2, x, y);
          T dh = deltas->data_at(0, b + 3, x, y);

          cv::Point_<T> center = rectangle::getRectangleCenter(anchors[index]);

          T cx = dx * anchors[index].width  + center.x;
          T cy = dy * anchors[index].height + center.y;
          T cw = std::exp(dw) * anchors[index].width;
          T ch = std::exp(dh) * anchors[index].height;

          proposals.push_back(rectangle::centeredRectangle(cx, cy, cw, ch)); ++index;
        }
      }
    }

    return proposals;
  }

  /** \brief Filters out unimportant proposals, by applying the non-maximum suppression algorithm.
   *  \param [in]  proposals   Vector of proposal rectangles.
   *  \param [in]  scores      Vector of proposal scores.
   *  \param [in]  threshold   Threshold to be used for filtering.
   *  \return Vector of indices corresponding to the proposals to be kept.
   */
  template <typename T>
  std::vector<size_t> applyNonMaximumSuppression(std::vector<cv::Rect_<T>> const & proposals,
                                                 std::vector<T>            const & scores,
                                                 float                     const   threshold)
  {
    std::vector<size_t> indices, order;

    order.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) { order.push_back(i); }

    while (!order.empty()) {
      indices.push_back(order[0]);

      std::vector<T> overlap;
      overlap.reserve(order.size() - 1);

      for (size_t i = 1; i < order.size(); ++i) {
        T area = (proposals[order[0]] & proposals[order[i]]).area();
        overlap.push_back(area / (proposals[order[0]].area() + proposals[order[i]].area() - area));
      }

      std::vector<size_t> new_order;
      for (size_t i = 0; i < overlap.size(); ++i) {
        if (overlap[i] <= threshold) { new_order.push_back(order[i + 1]); }
      }

      order = new_order;
    }

    return indices;
  }
}
}
}

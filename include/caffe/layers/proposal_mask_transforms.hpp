#pragma once

#include <opencv2/opencv.hpp>
#include <cassert>

namespace caffe {
namespace proposal_layer {
namespace algorithms {

  /** \brief Calculates the overlap of two input masks.
   *  \param [in]  box1        Input box1.
   *  \param [in]  box2        Input box2.
   *  \param [in]  mask1       Input mask1.
   *  \param [in]  mask2       Input mask2.
   *  \return overlap.
   */
  template <typename T>
  float maskOverlap(cv::Mat_<T> const & box1,
                    cv::Mat_<T> const & box2,
                    cv::Mat     const & mask1,
                    cv::Mat_<T> const & mask2
  ) {

    auto x1 = std::max(box1.template at<T>(0), box2.template at<T>(0));
    auto y1 = std::max(box1.template at<T>(1), box2.template at<T>(1));
    auto x2 = std::min(box1.template at<T>(2), box2.template at<T>(2));
    auto y2 = std::min(box1.template at<T>(3), box2.template at<T>(3));

    if ((x1 > x2) || (y1 > y2))
      return 0.;

    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;

    // Get masks in the intersection part
    int start_ya = y1 - box1.template at<T>(1);
    int start_xa = x1 - box1.template at<T>(0);
    auto inter_maska = mask1( cv::Rect( start_xa, start_ya, w, h));

    int start_yb = y1 - box2.template at<T>(1);
    int start_xb = x1 - box2.template at<T>(0);
    auto inter_maskb = mask2( cv::Rect( start_xb, start_yb, w, h));

    assert(inter_maska.size() == inter_maskb.size());

    int inter = cv::countNonZero((inter_maska & inter_maskb));
    int union_set = cv::countNonZero(mask1) + cv::countNonZero(mask2) - inter;

    if (union_set < 1.0) return 0.;
    return float(inter) / float(union_set);
  }

  /** \brief Calculates the intersection of an external box and ground truth box
             and masks it according to gt_mask.
   *  \param [in]  ex_box      Input external box.
   *  \param [in]  gt_box      Input ground truth box.
   *  \param [in]  mask        Input mask.
   *  \param [in]  mask_size   Mask size (after resizing).
   *  \param [in]  threshold   Binarization threshold.
   *  \return Regression targets.
   */
  template <typename T>
  cv::Mat_<T> intersectMask(cv::Rect_<T> const & ex_box,
                            cv::Rect_<T> const & gt_box,
                            cv::Mat_<T>  const & mask,
                            int          const   mask_size,
                            float        const   threshold)
  {
    int x1 = std::round(std::max(ex_box.tl().x, gt_box.tl().x));
    int y1 = std::round(std::max(ex_box.tl().y, gt_box.tl().y));
    int x2 = std::round(std::min(ex_box.br().x, gt_box.br().x));
    int y2 = std::round(std::min(ex_box.br().y, gt_box.br().y));

    if (x1 > x2 || y1 > y2) { return cv::Mat_<T>::zeros(mask_size, mask_size); }

    cv::Point ex_box_tl(std::round(x1 - ex_box.tl().x), std::round(y1 - ex_box.tl().y));
    cv::Point gt_box_tl(std::round(x1 - gt_box.tl().x), std::round(y1 - gt_box.tl().y));
    cv::Size size      (x2 - x1 + 1, y2 - y1 + 1);

    cv::Rect boundaries = {0, 0, mask.cols, mask.rows};
    cv::Rect roi        = cv::Rect(gt_box_tl.x, gt_box_tl.y, size.width, size.height) & boundaries;
    cv::Mat_<T> intersect_mask = mask(roi);

    cv::Size regression_size(std::round(ex_box.br().x - ex_box.tl().x + 1),
                             std::round(ex_box.br().y - ex_box.tl().y + 1));

    cv::Mat_<T> regression  = cv::Mat_<T>::zeros(regression_size);
    boundaries              = {0, 0, regression.cols, regression.rows};
    roi                     = cv::Rect(ex_box_tl.x, ex_box_tl.y, size.width, size.height);
    cv::Rect new_boundaries = roi | boundaries;

    regression.create(new_boundaries.height, new_boundaries.width);
    intersect_mask.copyTo(regression(roi));
    cv::resize(regression, regression, cv::Size(mask_size, mask_size));

    return regression >= threshold;
  }

}
}
}

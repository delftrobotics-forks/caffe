#pragma once

#include <opencv2/opencv.hpp>

namespace caffe {
namespace proposal_layer {
namespace algorithms {

  /** \brief Calculates the intersection of an external box and ground truth box
             and masks it according to gt_mask.
   *  \param [in]  external    Input external box.
   *  \param [in]  annotated   Input ground truth box.
   *  \param [in]  mask        Input mask.
   *  \param [in]  mask_size   Mask size (after resizing).
   *  \param [in]  threshold   Binarization threshold.
   *  \return Regression targets.
   */
  template <typename T>
  cv::Mat_<T> intersectMask(cv::Rect_<T> const & external,
                            cv::Rect_<T> const & annotated,
                            cv::Mat_<T>  const & mask,
                            int          const   mask_size,
                            float        const   threshold)
  {
    int x1 = std::round(std::max(external.tl().x, annotated.tl().x));
    int y1 = std::round(std::max(external.tl().y, annotated.tl().y));
    int x2 = std::round(std::min(external.br().x, annotated.br().x));
    int y2 = std::round(std::min(external.br().y, annotated.br().y));

    if (x1 > x2 || y1 > y2) { return cv::Mat_<T>::zeros(mask_size, mask_size); }

    cv::Point external_tl (std::round(x1 -  external.tl().x), std::round(y1 -  external.tl().y));
    cv::Point annotated_tl(std::round(x1 - annotated.tl().x), std::round(y1 - annotated.tl().y));
    cv::Size size(x2 - x1 + 1, y2 - y1 + 1);

    cv::Mat_<T> intersect_mask = mask(cv::Rect(annotated_tl.x, annotated_tl.y, size.width, size.height));

    cv::Size regression_size(std::round(external.br().x - external.tl().x + 1),
                             std::round(external.br().y - external.tl().y + 1));

    cv::Mat_<T> regression = cv::Mat_<T>::zeros(regression_size);
    intersect_mask.copyTo(regression(cv::Rect(external_tl.x, external_tl.y, size.width, size.height)));
    cv::resize(regression, regression, cv::Size(mask_size, mask_size));

    return regression >= threshold;
  }

}
}
}

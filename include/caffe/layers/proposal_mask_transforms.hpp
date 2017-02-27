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
   *  \param [in]  gt_mask        Input mask.
   *  \param [in]  mask_size   Mask size (after resizing).
   *  \param [in]  threshold   Binarization threshold.
   *  \return Regression targets.
   */
  template <typename T>
  cv::Mat intersectMask(cv::Mat const & ex_box,
                        cv::Mat const & gt_box,
                        cv::Mat const & gt_mask,
                        int     const   mask_size,
                        float   const   threshold)
  {
    assert(ex_box.rows == 1 && ex_box.cols == 4);
    assert(gt_box.rows == 1 && gt_box.cols == 4);

    T x1 = std::round(std::max(ex_box.at<T>(0, 0), gt_box.at<T>(0, 0)));
    T y1 = std::round(std::max(ex_box.at<T>(0, 1), gt_box.at<T>(0, 1)));
    T x2 = std::round(std::min(ex_box.at<T>(0, 2), gt_box.at<T>(0, 2)));
    T y2 = std::round(std::min(ex_box.at<T>(0, 3), gt_box.at<T>(0, 3)));

    if (x1 > x2 || y1 > y2) { return cv::Mat::zeros(mask_size, mask_size, CV_8U); }

    cv::Point_<T> ex_box_tl(x1 - ex_box.at<T>(0, 0), y1 - ex_box.at<T>(0, 1));
    cv::Point_<T> gt_box_tl(x1 - gt_box.at<T>(0, 0), y1 - gt_box.at<T>(0, 1));
    cv::Size_<T> size      (x2 - x1 + 1, y2 - y1 + 1);

    cv::Rect_<T> boundaries(0, 0, 0, 0);
    boundaries.height = gt_mask.dims == 2 ? gt_mask.rows : gt_mask.size[1];
    boundaries.width  = gt_mask.dims == 2 ? gt_mask.cols : gt_mask.size[2];

    cv::Rect_<T> roi = cv::Rect_<T>(gt_box_tl, size) & boundaries;

    cv::Mat inter_maskb(roi.height, roi.width, CV_8U);
    for (int i = roi.y; i < roi.y + roi.height; ++i) {
      for (int j = roi.x; j < roi.x + roi.width; ++j) {
        inter_maskb.at<uint8_t>(i - roi.y, j - roi.x) =
          gt_mask.dims == 2 ? uint8_t(gt_mask.at<T>(i, j)) : uint8_t(gt_mask.at<T>(0, i, j));
      }
    }

    cv::Size_<T> regression_size(ex_box.at<T>(0, 2) - ex_box.at<T>(0, 0) + 1,
                                 ex_box.at<T>(0, 3) - ex_box.at<T>(0, 1) + 1);

    cv::Mat regression = cv::Mat::zeros(regression_size, CV_32F);
    //boundaries         = {0.0, 0.0, T(regression.cols), T(regression.rows)};
    roi                = {ex_box_tl, size};
    //cv::Rect_<T> new_boundaries = roi | boundaries;
    //std::cout << ex_box_tl << " " << size << std::endl;
    //exit(0);

    //regression.create(std::round(new_boundaries.height), std::round(new_boundaries.width), regression.type());
    inter_maskb.copyTo(regression(cv::Rect_<T>(ex_box_tl, size)));
    cv::resize(regression, regression, cv::Size(mask_size, mask_size));

    return (regression >= threshold) / 255;
  }


  /** \brief Calculates the intersection of an external box and ground truth box
             and masks it according to gt_mask.
   *  \param [in]  ex_box      Input external box.
   *  \param [in]  gt_box      Input ground truth box.
   *  \param [in]  gt_mask        Input mask.
   *  \param [in]  mask_size   Mask size (after resizing).
   *  \param [in]  threshold   Binarization threshold.
   *  \return Regression targets.
   */
  template <typename T>
  cv::Mat intersectMask(cv::Rect_<T> const & ex_box,
                        cv::Rect_<T> const & gt_box,
                        cv::Mat      const & gt_mask,
                        int          const   mask_size,
                        float        const   threshold)
  {
    T x1 = std::round(std::max(ex_box.tl().x, gt_box.tl().x));
    T y1 = std::round(std::max(ex_box.tl().y, gt_box.tl().y));
    T x2 = std::round(std::min(ex_box.br().x, gt_box.br().x));
    T y2 = std::round(std::min(ex_box.br().y, gt_box.br().y));

    if (x1 > x2 || y1 > y2) { return cv::Mat::zeros(mask_size, mask_size, CV_8U); }

    cv::Point_<T> ex_box_tl(x1 - ex_box.tl().x, y1 - ex_box.tl().y);
    cv::Point_<T> gt_box_tl(x1 - gt_box.tl().x, y1 - gt_box.tl().y);
    cv::Size_<T> size      (x2 - x1 + 1, y2 - y1 + 1);

    cv::Rect_<T> boundaries(0, 0, 0, 0);
    boundaries.height = gt_mask.dims == 2 ? gt_mask.rows : gt_mask.size[1];
    boundaries.width  = gt_mask.dims == 2 ? gt_mask.cols : gt_mask.size[2];

    cv::Rect_<T> roi = cv::Rect_<T>(gt_box_tl, size) & boundaries;

    cv::Mat inter_maskb(roi.height, roi.width, CV_8U);
    for (int i = roi.y; i < roi.y + roi.height; ++i) {
      for (int j = roi.x; j < roi.x + roi.width; ++j) {
        inter_maskb.at<uint8_t>(i - roi.y, j - roi.x) =
          gt_mask.dims == 2 ? uint8_t(gt_mask.at<T>(i, j)) : uint8_t(gt_mask.at<T>(0, i, j));
      }
    }

    cv::Size_<T> regression_size(ex_box.br().x - ex_box.tl().x + 1, ex_box.br().y - ex_box.tl().y + 1);

    cv::Mat regression = cv::Mat::zeros(regression_size, CV_8U);
    boundaries         = {0.0, 0.0, T(regression.cols), T(regression.rows)};
    roi                = cv::Rect_<T>(ex_box_tl.x, ex_box_tl.y, size.width, size.height);
    cv::Rect_<T> new_boundaries = roi | boundaries;

    regression.create(std::round(new_boundaries.height), std::round(new_boundaries.width), regression.type());
    inter_maskb.copyTo(regression(roi));
    cv::resize(regression, regression, cv::Size(mask_size, mask_size));

    return (regression >= threshold) / 255;
  }

}
}
}

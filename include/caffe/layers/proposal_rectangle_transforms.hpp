#pragma once

#include <opencv2/opencv.hpp>

namespace caffe {
namespace proposal_layer {
namespace algorithms {
    /** \brief Expands the target data into the 4-of-4*K representation used
               by the network (i.e. only one class has non-zero targets).
     *  \param [out]  target_mat       Target matrix.
     *  \param [out]  in_weights       Bounding boxes inside weights.
     *  \param [in]   target_rects     Target rectangles.
     *  \param [in]   target_labels    Target labels.
     *  \param [in]   num_classes      Number of classes.
     *  \param [in]   weights          Vector of weights.
     */
  template <typename T>
  void getRegressionLabels(cv::Mat                         & target_mat,
                           cv::Mat                         & in_weights,
                           cv::Mat                   const & target_rects,
                           std::vector<int>          const & target_labels,
                           int                       const   num_classes,
                           cv::Mat                   const & weights)
  {
    assert(target_rects.rows == target_labels.size());

    target_mat = cv::Mat::zeros(target_labels.size(), 4 * num_classes, cv::DataType<T>::type);
    in_weights = cv::Mat::zeros(target_labels.size(), 4 * num_classes, cv::DataType<T>::type);

    for (size_t i = 0; i < target_labels.size(); ++i) {
      if (target_labels[i] > 0) {
        target_mat.at<T>(i, 4 * target_labels[i])     = target_rects.at<T>(i, 0);
        target_mat.at<T>(i, 4 * target_labels[i] + 1) = target_rects.at<T>(i, 1);
        target_mat.at<T>(i, 4 * target_labels[i] + 2) = target_rects.at<T>(i, 2);
        target_mat.at<T>(i, 4 * target_labels[i] + 3) = target_rects.at<T>(i, 3);

        in_weights.at<T>(i, 4 * target_labels[i])     = weights.at<T>(0, 0);
        in_weights.at<T>(i, 4 * target_labels[i] + 1) = weights.at<T>(0, 1);
        in_weights.at<T>(i, 4 * target_labels[i] + 2) = weights.at<T>(0, 2);
        in_weights.at<T>(i, 4 * target_labels[i] + 3) = weights.at<T>(0, 3);
      }
    }
  }
    /** \brief Expands the target data into the 4-of-4*K representation used
               by the network (i.e. only one class has non-zero targets).
     *  \param [out]  target_mat       Target matrix.
     *  \param [out]  in_weights       Bounding boxes inside weights.
     *  \param [in]   target_rects     Target rectangles.
     *  \param [in]   target_labels    Target labels.
     *  \param [in]   num_classes      Number of classes.
     *  \param [in]   weights          Vector of weights.
     */
  template <typename T>
  void getRegressionLabels(cv::Mat_<T>                     & target_mat,
                           cv::Mat_<T>                     & in_weights,
                           std::vector<cv::Rect_<T>> const & target_rects,
                           std::vector<int>          const & target_labels,
                           int                       const   num_classes,
                           cv::Rect_<T>              const & weights = {1.0, 1.0, 1.0, 1.0})
  {
    assert(target_rects.size() == target_labels.size());

    target_mat = cv::Mat_<T>::zeros(target_labels.size(), 4 * num_classes);
    in_weights = cv::Mat_<T>::zeros(target_labels.size(), 4 * num_classes);

    for (size_t i = 0; i < target_labels.size(); ++i) {
      if (target_labels[i] > 0) {
        target_mat.template at<T>(i, 4 * target_labels[i])     = target_rects[i].x;
        target_mat.template at<T>(i, 4 * target_labels[i] + 1) = target_rects[i].y;
        target_mat.template at<T>(i, 4 * target_labels[i] + 2) = target_rects[i].width;
        target_mat.template at<T>(i, 4 * target_labels[i] + 3) = target_rects[i].height;

        in_weights.template at<T>(i, 4 * target_labels[i])     = weights.x;
        in_weights.template at<T>(i, 4 * target_labels[i] + 1) = weights.y;
        in_weights.template at<T>(i, 4 * target_labels[i] + 2) = weights.width;
        in_weights.template at<T>(i, 4 * target_labels[i] + 3) = weights.height;
      }
    }
  }

  /** \brief Computes bounding box regression targets for an image.
   *  \param [in]  ex_rois     Regions of interest from external source (anchors or proposals).
   *  \param [in]  gt_rois     Ground truth regions of interest.
   *  \param [in]  means       Normalization mean.
   *  \param [in]  std         Normalization stdev.
   *  \param [in]  normalize   True if boxes are normalized, false otherwise.
   *  \return Vector with bounding box regression targets.
   */
  template <typename T>
  cv::Mat computeTargets(std::vector<cv::Mat> const & ex_rois,
                         std::vector<cv::Mat> const & gt_rois,
                         cv::Mat              const & mean,
                         cv::Mat              const & std,
                         bool normalize = true)
  {
    assert(ex_rois.size() == gt_rois.size());
    cv::Mat targets(gt_rois.size(), 4, cv::DataType<T>::type);

    for (size_t i = 0; i < ex_rois.size(); ++i) {
      T ex_width          = ex_rois[i].at<T>(0, 2) - ex_rois[i].at<T>(0, 0) + 1;
      T ex_height         = ex_rois[i].at<T>(0, 3) - ex_rois[i].at<T>(0, 1) + 1;
      T ex_x              = ex_rois[i].at<T>(0, 0) + 0.5 * ex_width;
      T ex_y              = ex_rois[i].at<T>(0, 1) + 0.5 * ex_height;

      T gt_width          = gt_rois[i].at<T>(0, 2) - gt_rois[i].at<T>(0, 0) + 1;
      T gt_height         = gt_rois[i].at<T>(0, 3) - gt_rois[i].at<T>(0, 1) + 1;
      T gt_x              = gt_rois[i].at<T>(0, 0) + 0.5 * gt_width;
      T gt_y              = gt_rois[i].at<T>(0, 1) + 0.5 * gt_height;

      targets.at<T>(i, 0) = (gt_x - ex_x) / ex_width;
      targets.at<T>(i, 1) = (gt_y - ex_y) / ex_height;
      targets.at<T>(i, 2) = std::log(gt_width / ex_width);
      targets.at<T>(i, 3) = std::log(gt_height / ex_height);

      if (normalize) {
        targets.at<T>(i, 0) = (targets.at<T>(i, 0) - mean.at<T>(0, 0)) / std.at<T>(0, 0);
        targets.at<T>(i, 1) = (targets.at<T>(i, 1) - mean.at<T>(0, 1)) / std.at<T>(0, 1);
        targets.at<T>(i, 2) = (targets.at<T>(i, 2) - mean.at<T>(0, 2)) / std.at<T>(0, 2);
        targets.at<T>(i, 3) = (targets.at<T>(i, 3) - mean.at<T>(0, 3)) / std.at<T>(0, 3);
      }
    }

    return targets;
  }


  /** \brief Computes bounding box regression targets for an image.
   *  \param [in]  ex_rois     Regions of interest from external source (anchors or proposals).
   *  \param [in]  gt_rois     Ground truth regions of interest.
   *  \param [in]  means       Normalization mean.
   *  \param [in]  std         Normalization stdev.
   *  \param [in]  normalize   True if boxes are normalized, false otherwise.
   *  \return Vector with bounding box regression targets.
   */
  template <typename T>
  std::vector<cv::Rect_<T>> computeTargets(std::vector<cv::Rect_<T>> const & ex_rois,
                                           std::vector<cv::Rect_<T>> const & gt_rois,
                                           cv::Rect_<T>              const & mean = {0.0, 0.0, 0.0, 0.0},
                                           cv::Rect_<T>              const & std  = {0.1, 0.1, 0.2, 0.2},
                                           bool normalize                         = true)
  {
    assert(ex_rois.size() == gt_rois.size());

    std::vector<cv::Rect_<T>> targets;
    targets.reserve(gt_rois.size());

    for (size_t i = 0; i < ex_rois.size(); ++i) {
      T ex_width      = ex_rois[i].br().x - ex_rois[i].tl().x + 1;
      T ex_height     = ex_rois[i].br().y - ex_rois[i].tl().y + 1;
      T ex_x          = ex_rois[i].tl().x + 0.5 * ex_width;
      T ex_y          = ex_rois[i].tl().y + 0.5 * ex_height;

      T gt_width      = gt_rois[i].br().x - gt_rois[i].tl().x + 1;
      T gt_height     = gt_rois[i].br().y - gt_rois[i].tl().y + 1;
      T gt_x          = gt_rois[i].tl().x + 0.5 * gt_width;
      T gt_y          = gt_rois[i].tl().y + 0.5 * gt_height;

      T target_x      = (gt_x - ex_x) / ex_width;
      T target_y      = (gt_y - ex_y) / ex_height;
      T target_width  = std::log(gt_width / ex_width);
      T target_height = std::log(gt_height / ex_height);

      targets.emplace_back(target_x, target_y, target_width, target_height);
    }

    if (normalize) {
      for (auto & t : targets) {
        t.x      = (t.x - mean.x) / std.x;
        t.y      = (t.y - mean.y) / std.y;
        t.width  = (t.width - mean.width) / std.width;
        t.height = (t.height - mean.height) / std.height;
      }
    }

    return targets;
  }

}
}
}

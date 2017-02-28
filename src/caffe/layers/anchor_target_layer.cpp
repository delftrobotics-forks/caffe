#include "caffe/layers/anchor_target_layer.hpp"
#include "caffe/layers/proposal_common.hpp"
#include "caffe/layers/proposal_anchor_transforms.hpp"
#include "caffe/layers/proposal_rectangle_transforms.hpp"
#include "caffe/util/rng.hpp"

#include <vector>

namespace {

/** \brief Filter rectangle based on image dimensions.
 *  \param [in,out]  rectangles     Vector of rectangles.
 *  \param [in]      dimensions     Input dimensions.
 *  \param [in]      allowed_border True if the dimensions of the rectangles will be changed, false otherwise.
 *  \return Vector of indices corresponding to the clipped rectangles.
 */
template <typename T>
std::vector<size_t> filterDimensions(std::vector<cv::Rect_<T>> const & rectangles,
                                     cv::Size                  const   dimensions,
                                     int                       const   allowed_border)
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < rectangles.size(); ++i) {
    cv::Point_<T> tl = rectangles[i].tl();
    cv::Point_<T> br = rectangles[i].br();

    if (tl.x >= -allowed_border && tl.y >= -allowed_border &&
        br.x <= dimensions.width + allowed_border && br.y <= dimensions.height + allowed_border) {
      indices.push_back(i);
    }
  }

  return indices;
}

}

namespace caffe {

/** \brief Implements the layer setup function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                          std::vector<Blob<Dtype>*> const & top)
{
  parameters_ = this->layer_param_.anchor_target_param();
  anchors_    = proposal_layer::anchor::generateBaseAnchors<Dtype>({ 0.5, 1, 2 }, { 8, 16, 32 }, 16);

  int width  = bottom[0]->shape(bottom[0]->num_axes() - 1);
  int height = bottom[0]->shape(bottom[0]->num_axes() - 2);

  // labels
  top[0]->Reshape({ 1, 1, static_cast<int>(anchors_.size() * height * width) });
  // bbox_targets
  top[1]->Reshape({ 1, static_cast<int>(4 * anchors_.size()), height, width });
  // bbox_inside_weights
  top[2]->Reshape({ 1, static_cast<int>(4 * anchors_.size()), height, width });
  // bbox_outside_weights
  top[3]->Reshape({ 1, static_cast<int>(4 * anchors_.size()), height, width });
}

/** \brief Implements the layer reshaping function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void AnchorTargetLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
                                       std::vector<Blob<Dtype>*> const & top)
{
}

/** \brief Implements the forward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                           std::vector<Blob<Dtype>*> const & top)
{
  // Algorithm:
  //
  // for each (H, W) location i
  //   generate 9 anchor boxes centered on cell i
  //   apply predicted transform deltas at cell i to each of the 9 anchors
  // filter out-of-image anchors
  // measure GT overlap

  using namespace proposal_layer;

  // only single item batches
  assert(bottom[0]->shape()[0] == 1);

  std::vector<cv::Rect_<Dtype>> gt_boxes = blob::extractRectsFromMatrix<Dtype>(*bottom[1], 0);

  layer_size_.height = bottom[0]->shape(bottom[0]->num_axes() - 2);
  layer_size_.width  = bottom[0]->shape(bottom[0]->num_axes() - 1);

  // Output target referenced value
  image_size_.height = bottom[2]->data_at({ 0, 0 });
  image_size_.width  = bottom[2]->data_at({ 0, 1 });

  // 1. Generate proposals from shifted anchors
  //    note: unlike proposal layer, in this stage, no deltas involved
  std::vector<cv::Rect_<Dtype>> all_anchors = anchor::generateShiftedAnchors<Dtype>(anchors_, layer_size_, parameters_.feat_stride());

  // only keep anchors inside the image
  std::vector<size_t> inds_inside           = ::filterDimensions(all_anchors, image_size_, parameters_.allowed_border());
  std::vector<cv::Rect_<Dtype>> anchors     = utils::select(all_anchors, inds_inside);

  // 2. For each anchor, we assign positive or negative
  // label: 1 is positive, 0 is negative, -1 is don't care
  std::vector<int> labels(inds_inside.size(), -1);

  // overlaps between the anchors and the gt boxes
  // overlaps (ex, gt)
  cv::Mat_<Dtype> overlaps = rectangle::intersectionOverUnion(anchors, gt_boxes);

  // find for each anchor the max overlaps
  std::vector<int> argmax_overlaps;
  std::vector<Dtype> max_overlaps;
  algorithms::max(max_overlaps, argmax_overlaps, overlaps, SearchType::COLUMNWISE);

  // find for each ground truth the max overlaps
  std::vector<int> gt_argmax_overlaps;
  std::vector<Dtype> gt_max_overlaps;
  algorithms::max(gt_max_overlaps, gt_argmax_overlaps, overlaps, SearchType::ROWWISE);

  if (!parameters_.clobber_positives()) {
    for (size_t i = 0; i < max_overlaps.size(); ++i) {
      // assign bg labels first so that positive labels can clobber them
      if (max_overlaps[i] < parameters_.negative_overlap()) {
        labels[i] = 0;
      }
    }
  }

  // We assign two types of anchors as positve
  // fg label: for each gt, anchor with highest overlap
  for (size_t i = 0; i < overlaps.rows; ++i) {
    for (size_t j = 0; j < overlaps.cols; ++j) {
      if (overlaps.template at<Dtype>(i, j) == gt_max_overlaps[j]) {
        labels[i] = 1;
      }
    }
  }

  // fg label: above threshold IOU
  for (size_t i = 0; i < max_overlaps.size(); ++i) {
    // assign bg labels first so that positive labels can clobber them
    if (max_overlaps[i] >= parameters_.positive_overlap()) {
      labels[i] = 1;
    }
  }


  if (parameters_.clobber_positives()) {
    for (size_t i = 0; i < max_overlaps.size(); ++i) {
      // assign bg labels last so that negative labels can clobber positives
      if (max_overlaps[i] < parameters_.negative_overlap()) {
        labels[i] = 0;
      }
    }
  }

  // find positive and negative labels
  std::vector<size_t> fg_inds;
  fg_inds.reserve(labels.size() / 2);
  std::vector<size_t> bg_inds;
  bg_inds.reserve(labels.size() / 2);
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == 1) fg_inds.push_back(i);
    if (labels[i] == 0) bg_inds.push_back(i);
  }

  // subsample positive labels if we have too many
  int num_fg = parameters_.fg_fraction() * parameters_.batchsize();
  if (fg_inds.size() > num_fg) {
    std::vector<size_t> disable_inds = algorithms::sampleWithoutReplacement(fg_inds, fg_inds.size() - num_fg);
    for (size_t const & id: disable_inds) {
      labels[id] = -1;
    }
  }

  // subsample negative labels if we have too many
  int num_bg = parameters_.batchsize();
  for (int const & l: labels) if (l == 1) num_bg--;
  if (bg_inds.size() > num_bg) {
    std::vector<size_t> disable_inds = algorithms::sampleWithoutReplacement(bg_inds, bg_inds.size() - num_bg);
    for (size_t const & id: disable_inds) {
      labels[id] = -1;
    }
  }

  // mix data from proposal_target_layer ?
  if (parameters_.mix_index()) {
    std::vector<size_t> bottom_fg = blob::extractVector<size_t>(*bottom[3], {0}, {bottom[3]->shape(0)});
    std::vector<size_t> bottom_bg = blob::extractVector<size_t>(*bottom[4], {0}, {bottom[4]->shape(0)});

    for (size_t const & id : bottom_fg) {
      auto it = std::find(inds_inside.begin(), inds_inside.end(), id);
      if (it != inds_inside.end()) {
        labels[it - inds_inside.begin()] = 1;
      }
    }

    for (size_t const & id : bottom_bg) {
      auto it = std::find(inds_inside.begin(), inds_inside.end(), id);
      if (it != inds_inside.end()) {
        labels[it - inds_inside.begin()] = 0;
      }
    }
  }

  std::vector<cv::Rect_<Dtype>> gt_max_overlaps_boxes;
  gt_max_overlaps_boxes.reserve(argmax_overlaps.size());
  for (int const & i: argmax_overlaps) {
    gt_max_overlaps_boxes.push_back(gt_boxes[i]);
  }

  // generate weights for boxes containing objects (label == 1)
  std::vector<cv::Rect_<Dtype>> bbox_targets = algorithms::computeTargets(anchors, gt_max_overlaps_boxes, {}, {}, false);
  std::vector<cv::Rect_<Dtype>> bbox_inside_weights;
  bbox_inside_weights.reserve(labels.size());
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == 1) {
      bbox_inside_weights.emplace_back(
        parameters_.bbox_inside_weights().x(),
        parameters_.bbox_inside_weights().y(),
        parameters_.bbox_inside_weights().width(),
        parameters_.bbox_inside_weights().height()
      );
    } else {
      bbox_inside_weights.emplace_back(0, 0, 0, 0);
    }
  }

  // generate weights for boxes outside of objects (label == 0)
  std::vector<cv::Rect_<Dtype>> bbox_outside_weights = std::vector<cv::Rect_<Dtype>>();
  bbox_outside_weights.reserve(inds_inside.size());
  Dtype positive_weight;
  Dtype negative_weight;
  if (parameters_.positive_weight() < 0) {
    // uniform weighting of examples (given non-uniform sampling)
    int num_examples = 0;
    for (int const & l: labels) {
      if (l == 0 || l == 1) num_examples++;
    }
    positive_weight = 1.0 / num_examples;
    negative_weight = 1.0 / num_examples;
  } else {
    int num_positive = 0;
    int num_negative = 0;
    for (int const & l: labels) {
      if (l == 0) num_negative++;
      if (l == 1) num_positive++;
    }

    assert(parameters_.positive_weight() > 0 && parameters_.positive_weight() < 1);
    positive_weight = parameters_.positive_weight() / num_positive;
    negative_weight = (1.0 - parameters_.positive_weight()) / num_negative;
  }
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] == 1) {
      bbox_outside_weights.emplace_back(positive_weight, positive_weight, positive_weight, positive_weight);
    } else if (labels[i] == 0) {
      bbox_outside_weights.emplace_back(negative_weight, negative_weight, negative_weight, negative_weight);
    } else {
      bbox_outside_weights.emplace_back(0, 0, 0, 0);
    }
  }

  // Currently all the indices are in the clipped index space
  // we map up to original set of anchors
  // In this process, we need to set clipped boxes as label -1, weights 0

  bbox_inside_weights  = utils::unmap<cv::Rect_<Dtype>>(bbox_inside_weights, all_anchors.size(), inds_inside, {0, 0, 0, 0});
  labels               = utils::unmap(labels, all_anchors.size(), inds_inside, -1);
  bbox_targets         = utils::unmap(bbox_targets, all_anchors.size(), inds_inside, {0, 0, 0, 0});
  bbox_outside_weights = utils::unmap(bbox_outside_weights, all_anchors.size(), inds_inside, {0, 0, 0, 0});

  // labels
  top[0]->Reshape({1, 1, static_cast<int>(anchors_.size() * layer_size_.height), layer_size_.width});
  Dtype * top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < labels.size(); ++i) { top_data[i] = labels[i]; }

  // bbox_targets
  top[1]->Reshape({1, static_cast<int>(anchors_.size() * 4), layer_size_.height, layer_size_.width});
  top_data = top[1]->mutable_cpu_data();
  for (size_t i = 0; i < bbox_targets.size(); ++i) {
    top_data[i * 4]     = bbox_targets[i].x;
    top_data[i * 4 + 1] = bbox_targets[i].y;
    top_data[i * 4 + 2] = bbox_targets[i].width;
    top_data[i * 4 + 3] = bbox_targets[i].height;
  }

  // bbox_inside_weights
  top[2]->Reshape({1, static_cast<int>(anchors_.size() * 4), layer_size_.height, layer_size_.width});
  top_data = top[2]->mutable_cpu_data();
  for (size_t i = 0; i < bbox_inside_weights.size(); ++i) {
    top_data[i * 4]     = bbox_inside_weights[i].x;
    top_data[i * 4 + 1] = bbox_inside_weights[i].y;
    top_data[i * 4 + 2] = bbox_inside_weights[i].width;
    top_data[i * 4 + 3] = bbox_inside_weights[i].height;
  }

  // bbox_outside_weights
  top[3]->Reshape({1, static_cast<int>(anchors_.size() * 4), layer_size_.height, layer_size_.width});
  top_data = top[3]->mutable_cpu_data();
  for (size_t i = 0; i < bbox_outside_weights.size(); ++i) {
    top_data[i * 4]     = bbox_outside_weights[i].x;
    top_data[i * 4 + 1] = bbox_outside_weights[i].y;
    top_data[i * 4 + 2] = bbox_outside_weights[i].width;
    top_data[i * 4 + 3] = bbox_outside_weights[i].height;
  }
}

// This layer doesn't have backward propagation
template <typename Dtype>
void AnchorTargetLayer<Dtype>::Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                                        std::vector<bool>         const & propagate_down,
                                        std::vector<Blob<Dtype>*> const & bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(AnchorTargetLayer);
#endif

INSTANTIATE_CLASS(AnchorTargetLayer);
REGISTER_LAYER_CLASS(AnchorTarget);

}

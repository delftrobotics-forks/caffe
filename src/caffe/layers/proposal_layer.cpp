#include "caffe/layers/proposal_layer.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>

#define DEBUG

namespace caffe {

using Rectangle = proposal_layer::Rectangle;
using Point     = proposal_layer::Point;

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                      std::vector<Blob<Dtype>*> const & top)
{
  auto const proposal_param = this->layer_param_.proposal_param();

  feat_stride_      = proposal_param.feat_stride();
  clip_denominator_ = proposal_param.clip_base();
  clip_threshold_   = 1.0 / clip_denominator_;
  use_clip_         = proposal_param.use_clip();

  top[0]->Reshape({ 1, 5 });

  generateReferenceAnchors(anchors_, { 0.5, 1, 2 }, { 8, 16, 32 }, 16);

  if (this->phase() == Phase::TRAIN) {
    top[1]->Reshape({ 1, 1 });
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
                                   std::vector<Blob<Dtype>*> const & top)
{
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                       std::vector<Blob<Dtype>*> const & top)
{
  assert(bottom[0]->shape()[0] == 1);

  layer_height_ = bottom[0]->shape()[2];
  layer_width_  = bottom[0]->shape()[3];

  int image_height = bottom[2]->data_at({ 0, 0 });
  int image_width  = bottom[2]->data_at({ 0, 1 });
  float ratio      = bottom[2]->data_at({ 0, 2 });

  size_t pre_nms_top_n  = 6000;
  size_t post_nms_top_n = 300;
  float nms_thresh      = 0.7;
  int min_size          = 16;

  std::vector<float> scores = generateScoresVector(*bottom[0], anchors_.size());

  std::vector<Rectangle> anchors;
  generateShiftedAnchors(anchors, anchors_, layer_width_, layer_height_, feat_stride_);
  anchor_index_before_clip_ = clipAnchors(anchors, image_width, image_height, false);

  std::vector<Rectangle> proposals;
  generateProposals(proposals, anchors, bottom[1]);

  proposal_index_before_clip_ = clipAnchors(proposals, image_width, image_height, true);
  ind_after_filter_           = getLargeRectangles(proposals, min_size * ratio);
  proposals                   = proposal_layer::select(proposals, ind_after_filter_);
  scores                      = proposal_layer::select(scores, ind_after_filter_);
  ind_after_sort_             = stableSort(scores);

  if (pre_nms_top_n > 0 && pre_nms_top_n < ind_after_sort_.size()) {
    ind_after_sort_.resize(pre_nms_top_n);
  }

  proposals       = proposal_layer::select(proposals, ind_after_sort_);
  scores          = proposal_layer::select(scores, ind_after_sort_);
  proposal_index_ = applyNonMaximumSuppression(proposals, scores, nms_thresh);

  if (post_nms_top_n > 0 && post_nms_top_n < proposal_index_.size()) {
    proposal_index_.resize(post_nms_top_n);
  }

  proposals = proposal_layer::select(proposals, proposal_index_);
  scores    = proposal_layer::select(scores, proposal_index_);

  top[0]->Reshape({ static_cast<int>(proposals.size()), 5 });
  Dtype * top_data = top[0]->mutable_cpu_data();

  for (size_t i = 0; i < proposals.size(); ++i) {
    Point top_left      = proposals[i].topLeft();
    Point bottom_right  = proposals[i].bottomRight();

    top_data[i * 5 + 0] = 0;
    top_data[i * 5 + 1] = top_left.x;
    top_data[i * 5 + 2] = top_left.y;
    top_data[i * 5 + 3] = bottom_right.x;
    top_data[i * 5 + 4] = bottom_right.y;
  }

  if (this->phase() == Phase::TRAIN) {
    top[1]->Reshape({ 1, static_cast<int>(proposals.size()) });
    Dtype * indices_data = top[1]->mutable_cpu_data();

    for (size_t i = 0; i < proposals.size(); ++i) {
      indices_data[i] = ind_after_filter_[ind_after_sort_[proposal_index_[i]]];
    }

  }
}

/** \brief Applies delta transformation on given anchors, to get transformed proposals.
 *  \param [out] proposals Vector of proposals.
 *  \param [in]  anchors   Vector of anchors.
 *  \param [in]  deltas    Blob with delta transformations.
 *  \return True if proposals are returned, false otherwise.
 */
template <typename Dtype>
bool ProposalLayer<Dtype>::generateProposals(std::vector<Rectangle>       & proposals,
                                             std::vector<Rectangle> const & anchors,
                                             Blob<Dtype>            const * deltas)
{
  if (anchors.empty()) { return false; }

  std::vector<int> shape = deltas->shape();
  proposals.resize(shape[1] * shape[2] * shape[3] / 4, Rectangle(0, 0, 0, 0));

  int index = 0;

  for (int x = 0; x < deltas->shape()[2]; ++x) {
    for (int y = 0; y < deltas->shape()[3]; ++y) {
      for (int b = 0; b < deltas->shape()[1]; b += 4) {
        Dtype dx = deltas->data_at(0, b,     x, y);
        Dtype dy = deltas->data_at(0, b + 1, x, y);
        Dtype dw = deltas->data_at(0, b + 2, x, y);
        Dtype dh = deltas->data_at(0, b + 3, x, y);

        Dtype cx = dx * anchors[index].width  + anchors[index].center.x;
        Dtype cy = dy * anchors[index].height + anchors[index].center.y;
        Dtype cw = std::exp(dw) * anchors[index].width;
        Dtype ch = std::exp(dh) * anchors[index].height;

        proposals[index++] = Rectangle(cx, cy, cw, ch);
      }
    }
  }

  return true;
}

template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::getLargeRectangles(std::vector<Rectangle> const & rectangles,
                                                             float                  const   min_size)
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < rectangles.size(); ++i) {
    Point top_left     = rectangles[i].topLeft();
    Point bottom_right = rectangles[i].bottomRight();

    if (bottom_right.x - top_left.x + 1 >= min_size && bottom_right.y - top_left.y + 1 >= min_size) {
      indices.push_back(i);
    }
  }

  return indices;
}

template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::clipAnchors(std::vector<Rectangle>       & rectangles,
                                                      int                    const   width,
                                                      int                    const   height,
                                                      bool                   const   auto_clip)
{
  std::vector<size_t> result;

  for (size_t i = 0; i < rectangles.size(); ++i) {
    Point top_left     = rectangles[i].topLeft();
    Point bottom_right = rectangles[i].bottomRight();

    bool in_range = top_left.x >= 0 && bottom_right.x <= width - 1 &&
                    top_left.y >= 0 && bottom_right.y <= height - 1;

    if (in_range) { result.push_back(i); }

    if (auto_clip) {
      top_left.x     = std::max(0.f, std::min(top_left.x,     static_cast<float>(width  - 1)));
      top_left.y     = std::max(0.f, std::min(top_left.y,     static_cast<float>(height - 1)));
      bottom_right.x = std::max(0.f, std::min(bottom_right.x, static_cast<float>(width  - 1)));
      bottom_right.y = std::max(0.f, std::min(bottom_right.y, static_cast<float>(height - 1)));

      rectangles[i] = Rectangle::fromCoordinates(top_left, bottom_right);
    }
  }

  return result;
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                                        std::vector<bool>         const & propagate_down,
                                        std::vector<Blob<Dtype>*> const & bottom)
{
  if (propagate_down[1]) {
    size_t num_elements = static_cast<size_t>(bottom[1]->count(0));
    Dtype * diff_data = bottom[1]->mutable_cpu_diff();
    memset(diff_data, 0, num_elements * sizeof(Dtype));

    std::vector<size_t> top_non_zero_ind;
    for (int i = 0; i < top[0]->shape()[0]; ++i) {
      for (int j = 0; j < top[0]->shape()[1]; ++j) {
        if (std::fabs(top[0]->diff_at({ i, j })) > 0) {
          top_non_zero_ind.push_back(i); break;
        }
      }
    }

    std::vector<size_t> unmap_val = proposal_layer::select(ind_after_filter_,
                                    proposal_layer::select(ind_after_sort_,
                                    proposal_layer::select(proposal_index_, top_non_zero_ind)));

    std::vector<bool> weight_out_proposal(unmap_val.size(), false);
    std::vector<bool> weight_out_anchor  (unmap_val.size(), false);

    // The following code needs to be optimized!
    // -------------------------------------------
    for (size_t i = 0; i < unmap_val.size(); ++i) {
      for (size_t j = 0; j < proposal_index_before_clip_.size(); ++j) {
        if (proposal_index_before_clip_[j] == unmap_val[i]) {
          weight_out_proposal[i] = true; break;
        }
      }
      for (size_t j = 0; j < anchor_index_before_clip_.size(); ++j) {
        if (anchor_index_before_clip_[j] == unmap_val[i]) {
          weight_out_anchor[i] = true; break;
        }
      }
    }
    // -------------------------------------------

    std::vector<size_t> channel(unmap_val.size());
    std::vector<size_t> width  (unmap_val.size());
    std::vector<size_t> height (unmap_val.size());

    for (size_t i = 0; i < unmap_val.size(); ++i) {
      channel[i] = unmap_val[i] % anchors_.size();
      width  [i] = std::fmod(static_cast<float>(unmap_val[i]) / anchors_.size(),  layer_width_);
      height [i] = std::fmod(static_cast<float>(unmap_val[i]) / anchors_.size() / layer_width_, layer_height_);
    }

    for (size_t i = 0; i < channel.size(); ++i) {
      Dtype dfdxc = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 1 });
      Dtype dfdyc = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 2 });
      Dtype dfdw  = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 3 });
      Dtype dfdh  = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 4 });

      int c = static_cast<int>(channel[i]);
      int h = static_cast<int>(height[i]);
      int w = static_cast<int>(width[i]);

      Dtype anchor_w = anchors_[c].width  / feat_stride_;
      Dtype anchor_h = anchors_[c].height / feat_stride_;

      size_t index = bottom[1]->offset(0, 4 * c, h, w);
      diff_data[index] = dfdxc * anchor_w * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 1, h, w);
      diff_data[index] = dfdyc * anchor_h * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 2, h, w);
      diff_data[index] = dfdw * std::exp(bottom[1]->data_at(0, 4 * c + 2, h, w)) * anchor_w * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 3, h, w);
      diff_data[index] = dfdh * std::exp(bottom[1]->data_at(0, 4 * c + 3, h, w)) * anchor_h * weight_out_proposal[i] * weight_out_anchor[i];
    }

    if (use_clip_) {
      for (size_t i = 0; i < channel.size(); ++i) {
        int c = static_cast<int>(channel[i]);
        int h = static_cast<int>(height[i]);
        int w = static_cast<int>(width[i]);

        for (int j = 0; j < 4; ++j) {
          size_t index = bottom[1]->offset(0, 4 * c + j, h, w);
          diff_data[index] = std::min(std::max(diff_data[index], -clip_threshold_), clip_threshold_);
        }
      }
    }
  }
}

/** \brief Generates anchors with different ratios and scales, starting from a
 *         square anchor with size \p base_size, centered at (0, 0).
 *  \param [out] anchors   Vector with the generated anchors.
 *  \param [in]  ratios    Vector with desired anchor ratios.
 *  \param [in]  scales    Vector with desired anchor scales.
 *  \param [in]  base_size Size of the (square) base anchor.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::generateReferenceAnchors(std::vector<Rectangle>       & anchors,
                                                    std::vector<float>     const & ratios,
                                                    std::vector<float>     const & scales,
                                                    int                    const   base_size)
{
  Rectangle base_anchor = Rectangle::fromCoordinates(Point(0, 0), Point(base_size - 1, base_size - 1));

  anchors.resize(ratios.size() * scales.size(), Rectangle(0, 0, 0, 0));

  int index = 0;
  for (float const & ratio : ratios) {
    Rectangle ratio_anchor = generateAnchorByRatio(base_anchor, ratio);
    for (float const & scale : scales) {
      anchors[index++] = generateAnchorByScale(ratio_anchor, scale);
    }
  }
}

/** \brief Shifts a set of reference anchors, such that they "cover" the whole surface
 *         of a WxH grid, where W and H are the width and height of the bottom layer.
 *  \param [out] shifted_anchors   Vector with the generated (shifted) anchors.
 *  \param [in]  reference_anchors Vector with the reference (input) anchors.
 *  \param [in]  layer_width       Width of the bottom layer.
 *  \param [in]  layer_height      Height of the bottom layer.
 *  \param [in]  feat_stride       Stride used when generating the shifted anchors.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::generateShiftedAnchors(std::vector<Rectangle>       & shifted_anchors,
                                                  std::vector<Rectangle> const & reference_anchors,
                                                  int                    const   layer_width,
                                                  int                    const   layer_height,
                                                  int                    const   feat_stride)
{
  shifted_anchors.resize(layer_width * layer_height * reference_anchors.size(), Rectangle(0, 0, 0, 0));

  int index = 0;

  for (int y = 0; y < layer_height; ++y) {
    for (int x = 0; x < layer_width; ++x) {
      for (Rectangle const & anchor : reference_anchors) {
        Point top_left     = anchor.topLeft();
        Point bottom_right = anchor.bottomRight();

        top_left.x = top_left.x + x * feat_stride;
        top_left.y = top_left.y + y * feat_stride;

        bottom_right.x = bottom_right.x + x * feat_stride;
        bottom_right.y = bottom_right.y + y * feat_stride;

        shifted_anchors[index++] = Rectangle::fromCoordinates(top_left, bottom_right);
      }
    }
  }
}

/** \brief Extracts the foreground scores from a blob and stores them in a vector of floats.
  * \param [in] blob   Blob that stores the scores.
  * \param [in] offset Position in the blob starting from which scores are stored.
  * \return Vector with scores, stored as float values.
  */
template <typename Dtype>
std::vector<float> ProposalLayer<Dtype>::generateScoresVector(Blob<Dtype> const & blob, int const offset) const {
  std::vector<float> scores;
  std::vector<int> shape = blob.shape();

  scores.resize(shape[2] * shape[3] * (shape[1] - offset));

  int index = 0;
  for (int j = 0; j < shape[2]; ++j) {
    for (int k = 0; k < shape[3]; ++k) {
      for (int i = offset; i < shape[1]; ++i) {
        scores[index++] = blob.data_at(0, i, j, k);
      }
    }
  }

  return scores;
}

/** \brief Scales up an reference anchor, by a given factor.
 *  \param [in] anchor Anchor that is used as reference (to be scaled up).
 *  \param [in] scale  Factor by which the reference anchor is scaled up.
 *  \return Scaled up anchor.
 */
template <typename Dtype>
Rectangle ProposalLayer<Dtype>::generateAnchorByScale(Rectangle const & anchor, float const scale) {
  return Rectangle(anchor.center.x, anchor.center.y, anchor.width * scale, anchor.height * scale);
}

/** \brief Generates an anchor with the same area as an reference one and a given aspect ratio.
 *  \param [in] anchor Anchor that is used as reference.
 *  \param [in] ratio  Aspect ratio of the output anchor.
 *  \return Anchor with the given aspect ratio and same (reference) area.
 */
template <typename Dtype>
Rectangle ProposalLayer<Dtype>::generateAnchorByRatio(Rectangle const & anchor, float const ratio) {
  int width  = std::round(std::sqrt(anchor.area() / ratio));
  int height = std::round(width * ratio);
  return Rectangle(anchor.center.x, anchor.center.y, width, height);
}

template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::applyNonMaximumSuppression(std::vector<Rectangle> const & proposals,
                                                                     std::vector<float>     const & scores,
                                                                     float                  const   threshold)
{
  std::vector<size_t> result;

  std::vector<size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);

  while (!order.empty()) {
    result.push_back(order[0]);

    std::vector<float> overlap;

    for (size_t i = 1; i < order.size(); ++i) {
      //float area = proposals[order[0]].intersect(proposals[order[i]]).area();
      float x1   = std::max(proposals[order[0]].topLeft().x,     proposals[order[i]].topLeft().x);
      float y1   = std::max(proposals[order[0]].topLeft().y,     proposals[order[i]].topLeft().y);
      float x2   = std::min(proposals[order[0]].bottomRight().x, proposals[order[i]].bottomRight().x);
      float y2   = std::min(proposals[order[0]].bottomRight().y, proposals[order[i]].bottomRight().y);
      float area = std::max(0.0, x2 - x1 + 1.0) * std::max(0.0, y2 - y1 + 1.0);

      overlap.push_back(area / (proposals[order[0]].area() + proposals[order[i]].area() - area));
    }

    std::vector<size_t> new_order;
    for (size_t i = 0; i < overlap.size(); ++i) {
      if (overlap[i] <= threshold) { new_order.push_back(order[i + 1]); }
    }

    order = new_order;
  }

  return result;
}

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}

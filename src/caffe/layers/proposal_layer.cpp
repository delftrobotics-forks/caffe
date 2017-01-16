#include "caffe/layers/proposal_layer.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>

namespace caffe {

/** \brief Implements the layer setup function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                      std::vector<Blob<Dtype>*> const & top)
{
  parameters_ = this->layer_param_.proposal_param();
  anchors_    = generateBaseAnchors({ 0.5, 1, 2 }, { 8, 16, 32 }, 16);

  top[0]->Reshape({ 1, 5 });

  if (this->phase() == Phase::TRAIN) { top[1]->Reshape({ 1, 1 }); }
}

/** \brief Implements the layer reshaping function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
                                   std::vector<Blob<Dtype>*> const & top)
{
}

/** \brief Implements the forward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                       std::vector<Blob<Dtype>*> const & top)
{
  assert(bottom[0]->shape()[0] == 1);

  layer_size_.height = bottom[0]->shape()[2];
  layer_size_.width  = bottom[0]->shape()[3];

  image_size_.height = bottom[2]->data_at({ 0, 0 });
  image_size_.width  = bottom[2]->data_at({ 0, 1 });
  float ratio        = bottom[2]->data_at({ 0, 2 });

  std::vector<Dtype> scores       = generateScoresVector(*bottom[0], anchors_.size());
  std::vector<Rectanglef> anchors = generateShiftedAnchors(anchors_, layer_size_, parameters_.feat_stride());
  anchor_index_before_clip_       = clipRectangles(anchors, image_size_, false);

  std::vector<Rectanglef> proposals = generateProposals(anchors, bottom[1]);

  proposal_index_before_clip_ = clipRectangles(proposals, image_size_, true);
  ind_after_filter_           = getLargeRectangles(proposals, 16 * ratio);

  proposals       = proposal_layer::select(proposals, ind_after_filter_);
  scores          = proposal_layer::select(scores, ind_after_filter_);
  ind_after_sort_ = stableSort(scores);

  if (parameters_.top_pre_nms() > 0 && parameters_.top_pre_nms() < ind_after_sort_.size()) {
    ind_after_sort_.resize(parameters_.top_pre_nms());
  }

  proposals       = proposal_layer::select(proposals, ind_after_sort_);
  scores          = proposal_layer::select(scores, ind_after_sort_);
  proposal_index_ = applyNonMaximumSuppression(proposals, scores, parameters_.nms_thresh());

  if (parameters_.top_post_nms() > 0 && parameters_.top_post_nms() < proposal_index_.size()) {
    proposal_index_.resize(parameters_.top_post_nms());
  }

  proposals = proposal_layer::select(proposals, proposal_index_);
  scores    = proposal_layer::select(scores, proposal_index_);

  top[0]->Reshape({ static_cast<int>(proposals.size()), 5 });
  Dtype * top_data = top[0]->mutable_cpu_data();

  for (size_t i = 0; i < proposals.size(); ++i) {
    cv::Point2f tl = proposals[i].tl();
    cv::Point2f br = proposals[i].br();

    top_data[i * 5 + 0] = 0;
    top_data[i * 5 + 1] = tl.x;
    top_data[i * 5 + 2] = tl.y;
    top_data[i * 5 + 3] = br.x;
    top_data[i * 5 + 4] = br.y;
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
 *  \param [in]  anchors   Vector of anchors.
 *  \param [in]  deltas    Blob with delta transformations.
 *  \return Vector of proposals.
 */
template <typename Dtype>
std::vector<Rectanglef> ProposalLayer<Dtype>::generateProposals(std::vector<Rectanglef> const & anchors,
                                                                Blob<Dtype>             const * deltas)
{
  std::vector<int> shape = deltas->shape();
  std::vector<Rectanglef> proposals;

  int index = 0;

  proposals.reserve(shape[1] * shape[2] * shape[3] / 4);
  for (int x = 0; x < deltas->shape()[2]; ++x) {
    for (int y = 0; y < deltas->shape()[3]; ++y) {
      for (int b = 0; b < deltas->shape()[1]; b += 4) {
        Dtype dx = deltas->data_at(0, b,     x, y);
        Dtype dy = deltas->data_at(0, b + 1, x, y);
        Dtype dw = deltas->data_at(0, b + 2, x, y);
        Dtype dh = deltas->data_at(0, b + 3, x, y);

        Dtype cx = dx * anchors[index].width  + anchors[index].center().x;
        Dtype cy = dy * anchors[index].height + anchors[index].center().y;
        Dtype cw = std::exp(dw) * anchors[index].width;
        Dtype ch = std::exp(dh) * anchors[index].height;

        proposals.push_back(Rectanglef::centered(cx, cy, cw, ch)); ++index;
      }
    }
  }

  return proposals;
}

/** \brief Filters out rectangles with width or height lower than a given threshold.
 *  \param [in]  rectangles   Vector of rectangles.
 *  \param [in]  min_size     Threshold to be used for filtering.
 *  \return Vector of indices corresponding to the filtered rectangles.
 */
template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::getLargeRectangles(std::vector<Rectanglef> const & rectangles,
                                                             float                   const   min_size) const
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < rectangles.size(); ++i) {
    if (rectangles[i].width >= min_size && rectangles[i].height >= min_size) { indices.push_back(i); }
  }

  return indices;
}

/** \brief Clip rectangle dimensions to a given image size.
 *  \param [in,out]  rectangles   Vector of rectangles.
 *  \param [in]      image_size   Input image size.
 *  \param [in]      auto_clip    True if the dimensions of the rectangles will be changed, false otherwise.
 *  \return Vector of indices corresponding to the clipped rectangles.
 */
template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::clipRectangles(std::vector<Rectanglef>       & rectangles,
                                                         cv::Size                const   image_size,
                                                         bool                    const   auto_clip) const
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < rectangles.size(); ++i) {
    cv::Point2f tl = rectangles[i].tl();
    cv::Point2f br = rectangles[i].br();

    if (tl.x >= 0 && br.x <= image_size.width && tl.y >= 0 && br.y <= image_size.height) {
      indices.push_back(i);
    }

    if (auto_clip) {
      rectangles[i] &= Rectanglef(0, 0, image_size.width, image_size.height);
    }
  }

  return indices;
}

/** \brief Implements the backward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
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

    std::vector<bool> weight_out_proposal;
    std::vector<bool> weight_out_anchor;

    weight_out_proposal.reserve(unmap_val.size());
    weight_out_anchor.reserve(unmap_val.size());

    for (size_t i = 0; i < unmap_val.size(); ++i) {
      weight_out_proposal.push_back(std::binary_search(proposal_index_before_clip_.begin(),
                                                       proposal_index_before_clip_.end(), unmap_val[i]));
      weight_out_anchor.push_back  (std::binary_search(anchor_index_before_clip_.begin(),
                                                       anchor_index_before_clip_.end(), unmap_val[i]));
    }

    std::vector<size_t> channel(unmap_val.size());
    std::vector<size_t> width  (unmap_val.size());
    std::vector<size_t> height (unmap_val.size());

    for (size_t i = 0; i < unmap_val.size(); ++i) {
      channel[i] = unmap_val[i] % anchors_.size();
      width  [i] = std::fmod(static_cast<float>(unmap_val[i]) / anchors_.size(),  layer_size_.width);
      height [i] = std::fmod(static_cast<float>(unmap_val[i]) / anchors_.size() / layer_size_.width, layer_size_.height);
    }

    for (size_t i = 0; i < channel.size(); ++i) {
      Dtype dfdxc = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 1 });
      Dtype dfdyc = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 2 });
      Dtype dfdw  = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 3 });
      Dtype dfdh  = top[0]->diff_at({ static_cast<int>(top_non_zero_ind[i]), 4 });

      int c = static_cast<int>(channel[i]);
      int h = static_cast<int>(height[i]);
      int w = static_cast<int>(width[i]);

      Dtype anchor_w = anchors_[c].width  / parameters_.feat_stride();
      Dtype anchor_h = anchors_[c].height / parameters_.feat_stride();

      size_t index = bottom[1]->offset(0, 4 * c, h, w);
      diff_data[index] = dfdxc * anchor_w * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 1, h, w);
      diff_data[index] = dfdyc * anchor_h * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 2, h, w);
      diff_data[index] = dfdw * std::exp(bottom[1]->data_at(0, 4 * c + 2, h, w)) * anchor_w * weight_out_proposal[i] * weight_out_anchor[i];

      index = bottom[1]->offset(0, 4 * c + 3, h, w);
      diff_data[index] = dfdh * std::exp(bottom[1]->data_at(0, 4 * c + 3, h, w)) * anchor_h * weight_out_proposal[i] * weight_out_anchor[i];
    }

    if (parameters_.use_clip()) {
      for (size_t i = 0; i < channel.size(); ++i) {
        int c = static_cast<int>(channel[i]);
        int h = static_cast<int>(height[i]);
        int w = static_cast<int>(width[i]);

        for (int j = 0; j < 4; ++j) {
          size_t index = bottom[1]->offset(0, 4 * c + j, h, w);

          Dtype clip_threshold = 1.0 / parameters_.clip_base();
          diff_data[index] = std::min(std::max(diff_data[index], -clip_threshold), clip_threshold);
        }
      }
    }
  }
}

/** \brief Generates anchors with different ratios and scales, starting from a
 *         square anchor with size 'base_size', centered at (0, 0).
 *  \param [in]   ratios      Vector with desired anchor ratios.
 *  \param [in]   scales      Vector with desired anchor scales.
 *  \param [in]   base_size   Size of the (square) base anchor.
 *  \return Vector with the generated anchors.
 */
template <typename Dtype>
std::vector<Rectanglef> ProposalLayer<Dtype>::generateBaseAnchors(std::vector<float> const & ratios,
                                                                  std::vector<float> const & scales,
                                                                  int                const   base_size)
{
  Rectanglef base_anchor(0, 0, base_size, base_size);

  std::vector<Rectanglef> anchors;
  anchors.reserve(ratios.size() * scales.size());

  for (auto const ratio : ratios) {
    Rectanglef ratio_anchor = generateAnchorByRatio(base_anchor, ratio);
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
template <typename Dtype>
std::vector<Rectanglef> ProposalLayer<Dtype>::generateShiftedAnchors(std::vector<Rectanglef> const & base_anchors,
                                                                     cv::Size                const   layer_size,
                                                                     int                     const   feat_stride)
{
  std::vector<Rectanglef> anchors;
  anchors.reserve(layer_size.width * layer_size.height * base_anchors.size());

  for (int y = 0; y < layer_size.height; ++y) {
    for (int x = 0; x < layer_size.width; ++x) {
      for (auto const & anchor : base_anchors) {
        cv::Point2f tl = anchor.tl() + cv::Point2f(x * feat_stride, y * feat_stride);
        cv::Point2f br = anchor.br() + cv::Point2f(x * feat_stride, y * feat_stride);
        anchors.push_back({ tl, br });
      }
    }
  }

  return anchors;
}

/** \brief Extracts the foreground scores from a blob and stores them in a vector of floats.
  * \param [in]  blob     Blob that stores the scores.
  * \param [in]  offset   Position in the blob starting from which scores are stored.
  * \return Vector with scores, stored as float values.
  */
template <typename Dtype>
std::vector<Dtype> ProposalLayer<Dtype>::generateScoresVector(Blob<Dtype> const & blob, int const offset) const {
  std::vector<Dtype> scores;
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
 *  \param [in]  anchor   Anchor that is used as reference (to be scaled up).
 *  \param [in]  scale    Factor by which the reference anchor is scaled up.
 *  \return Scaled up anchor.
 */
template <typename Dtype>
Rectanglef ProposalLayer<Dtype>::generateAnchorByScale(Rectanglef const & anchor, float const scale) const {
  cv::Point center = anchor.center();
  return Rectanglef::centered(center.x, center.y, anchor.width * scale, anchor.height * scale);
}

/** \brief Generates an anchor with the same area as an reference one and a given aspect ratio.
 *  \param [in]  anchor   Anchor that is used as reference.
 *  \param [in]  ratio    Aspect ratio of the output anchor.
 *  \return Anchor with the given aspect ratio and same (reference) area.
 */
template <typename Dtype>
Rectanglef ProposalLayer<Dtype>::generateAnchorByRatio(Rectanglef const & anchor, float const ratio) const {
  cv::Point2f center = anchor.center();
  float   width      = std::round(std::sqrt(anchor.area() / ratio));
  cv::Size2f size    = cv::Size2f(width, std::round(width * ratio));

  return Rectanglef::centered(center.x, center.y, size.width, size.height);
}

/** \brief Filters out unimportant proposals, by applying the non-maximum suppression algorithm.
 *  \param [in]  proposals   Vector of proposal rectangles.
 *  \param [in]  scores      Vector of proposal scores.
 *  \param [in]  threshold   Threshold to be used for filtering.
 *  \return Vector of indices corresponding to the proposals to be kept.
 */
template <typename Dtype>
std::vector<size_t> ProposalLayer<Dtype>::applyNonMaximumSuppression(std::vector<Rectanglef> const & proposals,
                                                                     std::vector<Dtype>      const & scores,
                                                                     float                   const   threshold) const
{
  std::vector<size_t> indices, order;

  order.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) { order.push_back(i); }

  while (!order.empty()) {
    indices.push_back(order[0]);

    std::vector<Dtype> overlap;
    overlap.reserve(order.size() - 1);

    for (size_t i = 1; i < order.size(); ++i) {
      Dtype area = (proposals[order[0]] & proposals[order[i]]).area();
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

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}

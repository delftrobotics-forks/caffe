#include "caffe/layers/proposal_layer.hpp"

#include <cfloat>
#include <cmath>
#include <limits>

namespace caffe {

/** \brief Implements the layer setup function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                      std::vector<Blob<Dtype>*> const & top)
{
  (void) bottom;

  parameters_ = this->layer_param_.proposal_param();
  anchors_    = proposal_layer::generateBaseAnchors<Dtype>({ 0.5, 1, 2 }, { 8, 16, 32 }, 16);

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
  (void) bottom;
  (void) top;
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
  Dtype ratio        = bottom[2]->data_at({ 0, 2 });

  auto scores                 = proposal_layer::generateScoresVector(*bottom[0], anchors_.size());
  auto anchors                = proposal_layer::generateShiftedAnchors(anchors_, layer_size_, parameters_.feat_stride());
  anchor_index_before_clip_   = proposal_layer::clipToDimensions(anchors, image_size_, false);

  auto proposals              = proposal_layer::generateProposals(anchors, bottom[1]);
  proposal_index_before_clip_ = proposal_layer::clipToDimensions(proposals, image_size_, true);
  ind_after_filter_           = proposal_layer::getLargeRectangles(proposals, 16 * ratio);

  proposals                   = proposal_layer::select(proposals, ind_after_filter_);
  scores                      = proposal_layer::select(scores, ind_after_filter_);
  ind_after_sort_             = proposal_layer::sort(scores);

  if (parameters_.top_pre_nms() > 0 && parameters_.top_pre_nms() < ind_after_sort_.size()) {
    ind_after_sort_.resize(parameters_.top_pre_nms());
  }

  proposals                   = proposal_layer::select(proposals, ind_after_sort_);
  scores                      = proposal_layer::select(scores, ind_after_sort_);
  proposal_index_             = proposal_layer::applyNonMaximumSuppression(proposals, scores, parameters_.nms_thresh());

  if (parameters_.top_post_nms() > 0 && parameters_.top_post_nms() < proposal_index_.size()) {
    proposal_index_.resize(parameters_.top_post_nms());
  }

  proposals                   = proposal_layer::select(proposals, proposal_index_);
  scores                      = proposal_layer::select(scores, proposal_index_);

  top[0]->Reshape({ static_cast<int>(proposals.size()), 5 });
  Dtype * top_data = top[0]->mutable_cpu_data();

  for (size_t i = 0; i < proposals.size(); ++i) {
    cv::Point_<Dtype> tl = proposals[i].tl();
    cv::Point_<Dtype> br = proposals[i].br();

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

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}

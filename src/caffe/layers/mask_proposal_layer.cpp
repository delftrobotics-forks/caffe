#include "caffe/layers/mask_proposal_layer.hpp"
#include "caffe/layers/proposal_anchor_transforms.hpp"
#include "caffe/layers/proposal_common.hpp"
#include "caffe/layers/proposal_mask_transforms.hpp"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <limits>

namespace caffe {

/** \brief Implements the layer setup function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void MaskProposalLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                          std::vector<Blob<Dtype>*> const & top)
{
  (void) bottom;
  parameters_ = this->layer_param_.mask_proposal_param();
  top[0]->Reshape({ 1, 1, parameters_.mask_size(), parameters_.mask_size() });

  if (this->phase() == Phase::TRAIN) { top[1]->Reshape({ 1, 1 }); }

}

/** \brief Implements the layer reshaping function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void MaskProposalLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
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
void MaskProposalLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                           std::vector<Blob<Dtype>*> const & top)
{
  std::vector<cv::Mat> resized_mask_pred;
  switch (this->phase()) {
    case Phase::TRAIN:
      ForwardTrain_cpu(bottom, top, resized_mask_pred);
      break;
    case Phase::TEST:
      //ForwardTest_cpu(bottom, top, resized_mask_pred);
      break;
    default:
      throw std::runtime_error("Phase should be either TRAIN or TEST.");
  }

  top[0]->Reshape(static_cast<int>(resized_mask_pred.size()), 1, parameters_.mask_size(), parameters_.mask_size());
  Dtype * top_data = top[0]->mutable_cpu_data();
  for (int c = 0; c < static_cast<int>(resized_mask_pred.size()); ++c) {
    for (int i = 0; i < resized_mask_pred[0].rows; ++i) {
      for (int j = 0; j < resized_mask_pred[0].cols; ++j) {
        int counter = top[0]->offset({c, 0, i, j});
        top_data[counter] = resized_mask_pred[c].at<Dtype>(i, j);
      }
    }
  }
}


/** \brief Implements the forward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void MaskProposalLayer<Dtype>::ForwardTrain_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                                std::vector<Blob<Dtype>*> const & top,
                                                std::vector<cv::Mat>            & resized_mask_pred)
{
  auto t1 = std::chrono::steady_clock::now();

  using namespace proposal_layer;
  auto mask_pred     = blob::extractMatrix<Dtype>(*bottom[0], { 0, 0 }, bottom[0]->shape());
  auto gt_masks      = blob::extractVectorOfMatrices<Dtype>(*bottom[1], { 0, 0, 0 }, bottom[1]->shape());
  auto gt_masks_info = blob::extractMatrix<Dtype>(*bottom[2], { 0, 0 }, bottom[2]->shape());
  auto num_mask_pred = mask_pred.rows;

  auto t2 = std::chrono::steady_clock::now();

  cv::Mat_<Dtype> top_label = cv::Mat_<Dtype>::zeros(gt_masks_info.rows, 1);

  for (std::size_t i = 0; i < num_mask_pred; ++i) {
    if (gt_masks_info.template at<Dtype>(i, 0) == -1) {
      top_label.template at<Dtype>(i, 0) = 0;
      continue;
    } else {
      auto info = gt_masks_info.row(i);
      auto gt_mask_intermediate = gt_masks.at(int(info.template at<Dtype>(0)));
      auto gt_mask = gt_mask_intermediate(cv::Rect(0, 0, info.template at<Dtype>(2), info.template at<Dtype>(1)));
      auto ex_mask = mask_pred.row(i).reshape(0, parameters_.mask_size());
      auto ex_box = info(cv::Rect(4, 0, 4, 1));
      auto gt_box = info(cv::Rect(8, 0, 4, 1));
      cv::resize(ex_mask, ex_mask, cv::Size(ex_box.template at<Dtype>(2) -  ex_box.template at<Dtype>(0) + 1,
                                            ex_box.template at<Dtype>(3) -  ex_box.template at<Dtype>(1) + 1));

      cv::threshold(ex_mask, ex_mask, parameters_.binarize_thresh(), 1, cv::THRESH_BINARY);

      float mask_overlap = proposal_layer::algorithms::maskOverlap(ex_box, gt_box, ex_mask, gt_mask);
      if (mask_overlap < parameters_.train_fg_seg_thresh()) {
        top_label.template at<Dtype>(i, 0) = 0;
      } else {
        top_label.template at<Dtype>(i, 0) = info.template at<Dtype>(3);
      }
    }
  }

  // Output continuous mask for MNC
  for (int i = 0; i < num_mask_pred; ++i) {
    cv::Mat resized = mask_pred.row(i).reshape(1, parameters_.mask_size());
    resized_mask_pred.push_back(resized);
  }

  cv::Mat locations = top_label != 0;
  cv::findNonZero(locations, locations);
  std::vector<cv::Mat> channels(2);
  cv::split(locations, channels);
  cv::Mat pos_sample;
  if (channels.size() > 0) {
    pos_sample = channels.back();
  }
  pos_sample.copyTo(pos_sample_);

  top[1]->Reshape({top_label.rows, 1});
  Dtype * top_data = top[1]->mutable_cpu_data();

  for (int i = 0; i < top_label.total(); ++i) {
    top_data[i] = top_label.template at<Dtype>(i);
  }

  auto t3 = std::chrono::steady_clock::now();
  std::cout << "Time spent: " << float((t2 - t1).count()) / (t3 - t1).count() << std::endl;
}

/** \brief Implements the forward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void MaskProposalLayer<Dtype>::ForwardTest_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                               std::vector<Blob<Dtype>*> const & top,
                                               cv::Mat & resized_mask_pred)
{
  std::cerr << "CURRENTLY IN FORWARD TEST CPU" << std::endl;
  using namespace proposal_layer;
  auto mask_pred     = blob::extractMatrix<Dtype>(*bottom[0], { 0, 0 }, bottom[0]->shape());
  auto num_mask_pred = mask_pred.rows;
  for (int i = 0; i < num_mask_pred; ++i) {
    cv::Mat resized = mask_pred.row(i).reshape(1, parameters_.mask_size());
    resized_mask_pred.push_back(resized);
  }
  std::cerr << "DONE" << std::endl;
}


/** \brief Implements the backward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void MaskProposalLayer<Dtype>::Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                                            std::vector<bool>         const & propagate_down,
                                            std::vector<Blob<Dtype>*> const & bottom)
{
  if (propagate_down.at(0)) {
    size_t num_elements = static_cast<size_t>(bottom[0]->count(0));
    Dtype * bottom_diff_data = bottom[0]->mutable_cpu_diff();
    memset(bottom_diff_data, 0, num_elements * sizeof(Dtype));

    for (int c = 0; c < pos_sample_.total(); ++c) {
      for (int i = 0; i < parameters_.mask_size(); ++i) {
        for (int j = 0; j < parameters_.mask_size(); ++j) {
          int counter = bottom[0]->offset({int(pos_sample_.at<uint>(c)), i*parameters_.mask_size()+j});
          bottom_diff_data[counter] = top[0]->diff_at({ int(pos_sample_.at<uint>(c)), 0, i, j});
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaskProposalLayer);
#endif

INSTANTIATE_CLASS(MaskProposalLayer);
REGISTER_LAYER_CLASS(MaskProposal);

}


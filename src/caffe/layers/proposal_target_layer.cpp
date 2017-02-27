#include "caffe/layers/proposal_target_layer.hpp"
#include "caffe/layers/proposal_common.hpp"
#include "caffe/layers/proposal_anchor_transforms.hpp"
#include "caffe/layers/proposal_mask_transforms.hpp"
#include "caffe/layers/proposal_rectangle_transforms.hpp"

#include <chrono>
#include <set>

namespace caffe {
namespace proposal_layer {

struct ROIParameters {
  float fraction;
  float lo_thresh;
  float hi_thresh;

  ROIParameters(float const f, float const lt, float const ht):
    fraction(f), lo_thresh(lt), hi_thresh(ht) {}
};

struct ROISamplingInput {
  cv::Mat                    all_rois;
  cv::Mat                    gt_rois;
  std::vector<int>           gt_labels;
  int                        rois_per_image;
  int                        num_classes;
  cv::Mat                    gt_masks;
  float                      image_scale;
  cv::Mat                    mask_info;
  std::vector<ROIParameters> fg_params;
  std::vector<ROIParameters> bg_params;
  int                        mask_size;
  float                      binarize_thresh;
};

struct ROISamplingOutput {
  std::vector<cv::Mat>       rois;
  std::vector<cv::Mat>       mask_weight;
  std::vector<cv::Mat>       pos_masks;
  cv::Mat                    target_mat;
  cv::Mat                    in_weights;
  cv::Mat                    out_weights;
  std::vector<int>           labels;
  cv::Mat                    top_mask_info;
  std::set<size_t>           fg_inds;
  std::set<size_t>           bg_inds;
  std::vector<size_t>        keep_inds;
};

template <typename T>
std::vector<size_t> thresholdOverlaps(std::vector<T> const & overlaps,
                                      float          const   lo_thresh,
                                      float          const   hi_thresh)
{
  std::vector<size_t> indices;

  for (size_t i = 0; i < overlaps.size(); ++i) {
    if (overlaps[i] >= lo_thresh && overlaps[i] <= hi_thresh) { indices.push_back(i); }
  }

  return indices;
}

template <typename T>
std::set<size_t> sampleIndices(std::vector<T>              const & overlaps,
                                std::vector<ROIParameters> const & params,
                                int                        const   num_rois)
{
  std::set<size_t> indices;

  for (auto const & p : params) {
    std::vector<size_t> thresh_indices = thresholdOverlaps(overlaps, p.lo_thresh, p.hi_thresh);
    //size_t num_current_rois = std::min(thresh_indices.size(), size_t(std::round(num_rois * p.fraction)));

    //if (!thresh_indices.empty()) { thresh_indices = sampleWithoutReplacement(thresh_indices, num_current_rois); }

    indices.insert(thresh_indices.begin(), thresh_indices.end());
  }

  return indices;
}

template <typename T>
void sampleRois(ROISamplingOutput & output, ROISamplingInput const & input) {
  std::vector<int> gt_assignment;
  std::vector<T> max_overlaps;

  cv::Mat all_boxes = input.all_rois(cv::Rect(1, 0, 4, input.all_rois.rows));
  cv::Mat gt_boxes  = input.gt_rois (cv::Rect(0, 0, 4, input.gt_rois.rows));
  cv::Mat overlaps  = rectangle::intersectionOverUnion<T>(all_boxes, gt_boxes);

  algorithms::max<T>(max_overlaps, gt_assignment, overlaps, SearchType::COLUMNWISE);

  output.fg_inds = sampleIndices(max_overlaps, input.fg_params, input.rois_per_image);
  output.bg_inds = sampleIndices(max_overlaps, input.bg_params, input.rois_per_image - output.fg_inds.size());

  output.keep_inds.reserve(output.fg_inds.size() + output.bg_inds.size());
  output.keep_inds.insert(output.keep_inds.end(), output.fg_inds.begin(), output.fg_inds.end());
  output.keep_inds.insert(output.keep_inds.end(), output.bg_inds.begin(), output.bg_inds.end());

  output.labels = utils::select(utils::select(input.gt_labels, gt_assignment), output.keep_inds);
  std::fill(output.labels.begin() + output.fg_inds.size(), output.labels.end(), 0);

  std::vector<cv::Mat> gts;

  output.rois.reserve(output.keep_inds.size());
  gts.reserve(output.keep_inds.size());

  for (auto const & i : output.keep_inds) { output.rois.push_back(all_boxes.row(i));              }
  for (auto const & i : output.keep_inds) { gts.push_back(gt_boxes.row(gt_assignment[i])); }

  cv::Mat mean        = cv::Mat::zeros(1, 4, cv::DataType<T>::type);
  cv::Mat std         = (cv::Mat_<T>(1, 4) << 0.1, 0.1, 0.2, 0.2);
  cv::Mat target_data = algorithms::computeTargets<T>(output.rois, gts, mean, std);
  cv::Mat weights     = cv::Mat::ones(1, 4, cv::DataType<T>::type);

  algorithms::getRegressionLabels<T>(output.target_mat, output.in_weights, target_data,
                                     output.labels, input.num_classes, weights);

  cv::threshold(output.in_weights, output.out_weights, 0, 1, CV_THRESH_BINARY);

  if (!input.mask_info.empty() && !input.gt_masks.empty()) {
    cv::Mat scaled_rois(output.rois.size(), 4, cv::DataType<T>::type);

    for (size_t i = 0; i < output.rois.size(); ++i) { 
      cv::Mat scaled_roi = output.rois[i] * (1.0 / input.image_scale);
      scaled_roi.copyTo(scaled_rois.row(i));
    }

    cv::Mat scaled_gt_boxes = gt_boxes * (1.0 / input.image_scale);

    output.pos_masks.reserve(output.keep_inds.size());
    output.mask_weight.reserve(output.rois.size());

    for (size_t i = 0; i < output.rois.size(); ++i) {
      if (i < output.fg_inds.size()) {
        output.mask_weight.emplace_back(cv::Mat::ones(input.mask_size, input.mask_size, cv::DataType<T>::type));
      } else {
        output.mask_weight.emplace_back(cv::Mat::zeros(input.mask_size, input.mask_size, cv::DataType<T>::type));
      }
    }

    output.top_mask_info = cv::Mat::zeros(output.keep_inds.size(), 12, cv::DataType<T>::type);
    output.top_mask_info(cv::Range(output.fg_inds.size(), output.top_mask_info.rows), cv::Range(0, output.top_mask_info.cols)).setTo(-1);

    std::set<size_t>::iterator it;
    for (it = output.fg_inds.begin(); it != output.fg_inds.end(); ++it) {
      int i = std::distance(output.fg_inds.begin(), it);
      int j = gt_assignment[*it];

      cv::Size gt_mask_info(input.mask_info.at<T>(j, 1), input.mask_info.at<T>(j, 0));

      cv::Range ranges[3] = {{j, j + 1}, {0, gt_mask_info.height}, {0, gt_mask_info.width}};
      cv::Mat gt_mask3(input.gt_masks, ranges);
      cv::Mat ex_box = scaled_rois.row(i).clone();
      cv::Mat gt_box = scaled_gt_boxes.row(j).clone();

      for (int k = 0; k < 4; ++k) {
        ex_box.at<T>(0, k) = std::round(ex_box.at<T>(0, k));
        gt_box.at<T>(0, k) = std::round(gt_box.at<T>(0, k));
      }

      output.pos_masks.push_back(algorithms::intersectMask<T>(
        ex_box, gt_box, gt_mask3, input.mask_size, input.binarize_thresh));

      output.top_mask_info.at<T>(i,  0) = gt_assignment[*it];
      output.top_mask_info.at<T>(i,  1) = gt_mask_info.height;
      output.top_mask_info.at<T>(i,  2) = gt_mask_info.width;
      output.top_mask_info.at<T>(i,  3) = output.labels[i];
      output.top_mask_info.at<T>(i,  4) = ex_box.at<T>(0, 0);
      output.top_mask_info.at<T>(i,  5) = ex_box.at<T>(0, 1);
      output.top_mask_info.at<T>(i,  6) = ex_box.at<T>(0, 2);
      output.top_mask_info.at<T>(i,  7) = ex_box.at<T>(0, 3);
      output.top_mask_info.at<T>(i,  8) = gt_box.at<T>(0, 0);
      output.top_mask_info.at<T>(i,  9) = gt_box.at<T>(0, 1);
      output.top_mask_info.at<T>(i, 10) = gt_box.at<T>(0, 2);
      output.top_mask_info.at<T>(i, 11) = gt_box.at<T>(0, 3);
    }

    while (output.pos_masks.size() != output.pos_masks.capacity()) {
      output.pos_masks.emplace_back(cv::Mat::zeros(input.mask_size, input.mask_size, CV_8U));
    }
  }
}

}

/** \brief Implements the layer setup function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalTargetLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                            std::vector<Blob<Dtype>*> const & top)
{
  params_  = this->layer_param_.proposal_target_param();
  anchors_ = proposal_layer::anchor::generateBaseAnchors<Dtype>({0.5, 1, 2}, {8, 16, 32}, 16);

  int num_classes = int(params_.num_classes());
  int mask_size   = int(params_.mask_size());

  top[0]->Reshape({1, 5});
  top[1]->Reshape({1, 1});
  top[2]->Reshape({1, num_classes * 4});
  top[3]->Reshape({1, num_classes * 4});
  top[4]->Reshape({1, num_classes * 4});

  if (params_.mnc_mode()) {
    top[5]->Reshape({1, 1, mask_size, mask_size});
    top[6]->Reshape({1, 1, mask_size, mask_size});
    top[7]->Reshape({1, 4 });

    if (params_.mix_index()) {
      top[8]->Reshape({1, 4});
      top[9]->Reshape({1, 4});
    }
  }
}

/** \brief Implements the layer reshaping function.
 *  \param [in]  bottom   Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas   Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalTargetLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
                                         std::vector<Blob<Dtype>*> const & top)
{
}

/** \brief Implements the forward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                             std::vector<Blob<Dtype>*> const & top)
{
  using namespace proposal_layer;

  ROISamplingInput  input;
  ROISamplingOutput output;

  input.all_rois  = blob::extract<Dtype>(*bottom[0], {0, 0}, bottom[0]->shape());
  input.gt_rois   = blob::extract<Dtype>(*bottom[1], {0, 0}, bottom[1]->shape());
  input.gt_labels = blob::extractVector<int, Dtype>(*bottom[1], {0, 4}, {bottom[1]->shape(0), 5});

  // Potentially dangerous, will likely modify data from bottom[0]!
  cv::Mat temp = cv::Mat::zeros(input.gt_rois.rows, 5, input.gt_rois.type());
  input.gt_rois(cv::Rect(0, 0, 4, input.gt_rois.rows)).copyTo(temp(cv::Rect(1, 0, 4, input.gt_rois.rows)));
  input.all_rois.push_back(temp);

  if (params_.mnc_mode()) {
    input.gt_masks  = blob::extract<Dtype>(*bottom[3], {0, 0, 0}, bottom[3]->shape());
    input.mask_info = blob::extract<Dtype>(*bottom[4], {0, 0},    bottom[4]->shape());
  }

  for (size_t i = 0; i < params_.fg_fraction_size(); ++i) {
    input.fg_params.push_back({params_.fg_fraction(i), params_.fg_lo_thresh(i), params_.fg_hi_thresh(i)}); }
  for (size_t i = 0; i < params_.bg_fraction_size(); ++i) {
    input.bg_params.push_back({params_.bg_fraction(i), params_.bg_lo_thresh(i), params_.bg_hi_thresh(i)}); }

  input.rois_per_image = params_.batch_size();
  input.num_classes = params_.num_classes();
  input.image_scale = bottom[2]->data_at({0, 2});
  input.mask_size = params_.mask_size();
  input.binarize_thresh = params_.binarize_thresh();

  sampleRois<Dtype>(output, input);
  keep_indices_ = params_.bp_all() ? output.keep_inds : utils::convert<size_t, size_t>(output.fg_inds);

  top[0]->Reshape({int(output.keep_inds.size()), 5});
  Dtype * top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < output.rois.size(); ++i) {
    int label = keep_indices_[i] >= input.all_rois.rows ? 0 : input.all_rois.at<Dtype>(keep_indices_[i], 0);
    top_data[i * 5]     = label;
    top_data[i * 5 + 1] = output.rois[i].at<Dtype>(0, 0);
    top_data[i * 5 + 2] = output.rois[i].at<Dtype>(0, 1);
    top_data[i * 5 + 3] = output.rois[i].at<Dtype>(0, 2);
    top_data[i * 5 + 4] = output.rois[i].at<Dtype>(0, 3);
  }

  blob::writeVector(*top[1], output.labels);
  blob::writeMatrix(*top[2], output.target_mat);
  blob::writeMatrix(*top[3], output.in_weights);
  blob::writeMatrix(*top[4], output.out_weights);

  if (!output.pos_masks.empty()) {
    top[5]->Reshape({int(output.pos_masks.size()), 1, output.pos_masks[0].rows, output.pos_masks[0].cols});
  } else {
    top[5]->Reshape({0, 1, 0, 0});
  }

  int offset = 0;

  top_data = top[5]->mutable_cpu_data();
  for (size_t i = 0; i < output.pos_masks.size(); ++i) {
    for (size_t j = 0; j < output.pos_masks[i].total(); ++j) {
      top_data[offset++] = output.pos_masks[i].at<uint8_t>(j);
    }
  }

  if (!output.mask_weight.empty()) {
    top[6]->Reshape({int(output.mask_weight.size()), 1, output.mask_weight[0].rows, output.mask_weight[0].cols});
  } else {
    top[6]->Reshape({0, 1, 0, 0});
  }

  offset = 0;

  top[6]->Reshape({int(output.mask_weight.size()), 1, output.mask_weight[0].rows, output.mask_weight[0].cols});
  top_data = top[6]->mutable_cpu_data();
  for (size_t i = 0; i < output.mask_weight.size(); ++i) {
    for (size_t j = 0; j < output.mask_weight[i].total(); ++j) {
      top_data[offset++] = output.mask_weight[i].at<Dtype>(j);
    }
  }

  blob::writeMatrix<Dtype, Dtype>(*top[7], output.top_mask_info);

  if (params_.mix_index()) {
    std::vector<int> all_rois_index = blob::extractVector<int, Dtype>(*bottom[5], {0, 0}, {1, bottom[5]->shape()[1]});
    std::set<size_t> lower = utils::compareLower(output.fg_inds, all_rois_index.size());

    blob::writeVector(*top[8], utils::select(all_rois_index, lower));
    blob::writeVector(*top[9], utils::select(all_rois_index, output.bg_inds));
  }
}

/** \brief Implements the backward propagation function.
 *  \param [in]  bottom    Bottom (previous/input) Caffe layers.
 *  \param [in]  deltas    Top (next/output) Caffe layers.
 */
template <typename Dtype>
void ProposalTargetLayer<Dtype>::Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                                              std::vector<bool>         const & propagate_down,
                                              std::vector<Blob<Dtype>*> const & bottom)
{
  if (propagate_down[0]) {
    size_t num_elements = size_t(bottom[0]->count(0));
    Dtype * diff_data = bottom[0]->mutable_cpu_diff();
    memset(diff_data, 0, num_elements * sizeof(Dtype));

    std::vector<size_t> valid_indices;
    for (size_t i = 0; i < keep_indices_.size(); ++i) {
      if (keep_indices_[i] < size_t(bottom[0]->shape(0))) { valid_indices.push_back(i); }
    }

    std::vector<size_t> valid_bot_indices = proposal_layer::utils::select(keep_indices_, valid_indices);

    diff_data = bottom[0]->mutable_cpu_diff();
    for (size_t i = 0; i < valid_bot_indices.size(); ++i) {
      for (int j = 0; j < bottom[0]->shape(1); ++j) {
        int index = bottom[0]->offset({int(valid_bot_indices[i]), j});
        diff_data[index] = top[0]->diff_at({int(valid_indices[i]), j});
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ProposalTargetLayer);
#endif

INSTANTIATE_CLASS(ProposalTargetLayer);
REGISTER_LAYER_CLASS(ProposalTarget);

}

#include "caffe/layers/proposal_target_layer.hpp"
#include "caffe/layers/proposal_common.hpp"
#include "caffe/layers/proposal_anchor_transforms.hpp"
#include "caffe/layers/proposal_mask_transforms.hpp"
#include "caffe/layers/proposal_rectangle_transforms.hpp"

#include <set>

namespace caffe {
namespace proposal_layer {

template <typename T>
void handleError(int index, int iteration, T * top_data, std::vector<int> t) {
  for (int i = 0; i < int(t.size()); i++) {
    if (std::fabs(top_data[i] - t[i]) > 1e-3) {
      std::cout << "iter: " << iteration << " top" << index << ": " << top_data[i] << " " << t[i] << std::endl << std::endl;

      utils::print(t);
      std::cout << std::endl;
      for (int j = 0; j < int(t.size()); ++j) { std::cout << top_data[j] << " "; }
      std::cout << std::endl;
      exit(0);
    }
  }
}

template <typename T>
void handleError(int index, int iteration, T * top_data, cv::Mat t) {
  for (int i = 0; i < t.cols; i++) {
    if (std::fabs(top_data[i] - t.at<double>(0, i)) > 1e-3) {
      std::cout << "iter: " << iteration << " top" << index << ": " << top_data[i] << " " << t.at<double>(0, i) << std::endl << std::endl;

      std::cout << t << std::endl << std::endl;
      for (int j = 0; j < t.cols; ++j) { std::cout << top_data[j] << " "; }
      std::cout << std::endl;
      exit(0);
    }
  }
}

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
  cv::Mat                    sampling_mean;
  cv::Mat                    sampling_std;
  cv::Mat                    regression_weights;
  cv::Mat                    gt_masks;
  cv::Mat                    mask_info;
  int                        rois_per_image;
  int                        num_classes;
  int                        mask_size;
  float                      binarize_thresh;
  float                      image_scale;
  std::vector<ROIParameters> fg_params;
  std::vector<ROIParameters> bg_params;
};

struct ROISamplingOutput {
  std::vector<cv::Mat>       rois;
  std::vector<cv::Mat>       mask_weight;
  std::vector<cv::Mat>       pos_masks;
  cv::Mat                    target_mat;
  cv::Mat                    in_weights;
  cv::Mat                    out_weights;
  cv::Mat                    top_mask_info;
  std::set<size_t>           fg_inds;
  std::set<size_t>           bg_inds;
  std::vector<size_t>        keep_inds;
  std::vector<int>           labels;
};

/** \brief Selects the indices of overlaps within a given range.
 *  \param  [in]  overlaps    Vector with overlaps.
 *  \param  [in]  lo_thresh   Low threshold.
 *  \oaram  [in]  hi_thresh   High threshold.
 *  \return Vector with indices of thresholded values.
 */
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

/** \brief Samples foreground or background indices, based on given thresholds.
 *  \param  [in]  overlaps   Vector with overlaps.
 *  \param  [in]  params     Vector with ROI parameters (background/foreground thresholds).
 *  \param  [in]  num_rois   Number of ROIs per image.
 *  \return Set with sampled background/foreground indices.
 */
template <typename T>
std::set<size_t> sampleIndices(std::vector<T>              const & overlaps,
                                std::vector<ROIParameters> const & params,
                                int                        const   num_rois)
{
  std::set<size_t> indices;

  for (auto const & p : params) {
    std::vector<size_t> thresh_indices = thresholdOverlaps(overlaps, p.lo_thresh, p.hi_thresh);
    size_t num_current_rois = std::min(thresh_indices.size(), size_t(std::round(num_rois * p.fraction)));

    if (!thresh_indices.empty()) { thresh_indices = algorithms::sampleWithoutReplacement(thresh_indices, num_current_rois); }

    indices.insert(thresh_indices.begin(), thresh_indices.end());
  }

  return indices;
}

/** \brief Creates the top blobs for the forward propagation function.
 *  \param  [out]  top         Top (output) blobs.
 *  \param  [in]   bottom      Bottom (input) blobs.
 *  \param  [in]   indices     Indices of the foreground ROIs.
 *  \param  [in]   output      Output from the sampleROIs function.
 *  \param  [in]   input       Input to the sampleROIs function.
 *  \param  [in]   mix_index   True if anchors used for RPN and later layer should be mixed.
 */
template <typename T>
void processTopBlobs(std::vector<Blob<T>*> const & top,
                     std::vector<Blob<T>*> const & bottom,
                     std::vector<size_t>   const & indices,
                     ROISamplingOutput     const & output,
                     ROISamplingInput      const & input,
                     bool                          mix_index,
                     int iteration)
{
  //cv::FileStorage fs("/home/mmorariu/dump/forward_data_" + std::to_string(iteration) + ".yaml", cv::FileStorage::READ);

  top[0]->Reshape({int(output.keep_inds.size()), 5});
  T * top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < output.rois.size(); ++i) {
    int label = indices[i] >= input.all_rois.rows ? 0 : input.all_rois.template at<T>(indices[i], 0);
    top_data[i * 5]     = label;
    top_data[i * 5 + 1] = output.rois[i].at<T>(0, 0);
    top_data[i * 5 + 2] = output.rois[i].at<T>(0, 1);
    top_data[i * 5 + 3] = output.rois[i].at<T>(0, 2);
    top_data[i * 5 + 4] = output.rois[i].at<T>(0, 3);
  }

  //cv::Mat t0;
  //fs["top0"] >> t0;
  //handleError(0, iteration, top[0]->mutable_cpu_data(), t0);

  blob::writeVector(*top[1], output.labels);

  //cv::Mat t1;
  //fs["top1"] >> t1;
  //handleError(1, iteration, top[1]->mutable_cpu_data(), t1);

  blob::writeMatrix(*top[2], output.target_mat);

  //cv::Mat t2;
  //fs["top2"] >> t2;
  //handleError(2, iteration, top[2]->mutable_cpu_data(), t2);

  blob::writeMatrix(*top[3], output.in_weights);

  //cv::Mat t3;
  //fs["top3"] >> t3;
  //handleError(3, iteration, top[3]->mutable_cpu_data(), t3);

  blob::writeMatrix(*top[4], output.out_weights);

  //cv::Mat t4;
  //fs["top4"] >> t4;
  //handleError(4, iteration, top[4]->mutable_cpu_data(), t4);

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

  //cv::Mat t5;
  //fs["top5"] >> t5;
  //handleError(5, iteration, top[5]->mutable_cpu_data(), t5);

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
      top_data[offset++] = output.mask_weight[i].at<T>(j);
    }
  }

  //cv::Mat t6;
  //fs["top6"] >> t6;
  //handleError(6, iteration, top[6]->mutable_cpu_data(), t6);

  blob::writeMatrix<T, double>(*top[7], output.top_mask_info);

  //cv::Mat t7;
  //fs["top7"] >> t7;
  //handleError(7, iteration, top[7]->mutable_cpu_data(), t7);

  if (mix_index) {
    std::vector<int> all_rois_index = blob::extractVector<int, T>(*bottom[5], {0, 0}, {1, bottom[5]->shape()[1]});
    std::set<size_t> lower = utils::compareLower(output.fg_inds, all_rois_index.size());

    auto t1 = utils::select(all_rois_index, lower);
    blob::writeVector(*top[8], t1);
    //handleError(8, iteration, top[8]->mutable_cpu_data(), t1);

    auto t2 = utils::select(all_rois_index, output.bg_inds);
    blob::writeVector(*top[9], t2);
    //handleError(9, iteration, top[9]->mutable_cpu_data(), t2);
  }

  //fs.close();
}

/** \brief Generates a random sample of ROIs comprising foreground and background examples.
 *  \param [out]  output   ROI sampling output.
 *  \param [in]   input    ROI sampling input.
 */
template <typename T>
void sampleROIs(ROISamplingOutput & output, ROISamplingInput const & input) {
  std::vector<int> gt_assignment;
  std::vector<T> max_overlaps;

  // Get the predicted boxes and the ground truth.
  cv::Mat all_boxes = input.all_rois(cv::Rect(1, 0, 4, input.all_rois.rows));
  cv::Mat gt_boxes  = input.gt_rois (cv::Rect(0, 0, 4, input.gt_rois.rows));

  // Compute intersection over union and max over that.
  cv::Mat overlaps  = rectangle::intersectionOverUnion<T>(all_boxes, gt_boxes);
  algorithms::max<T>(max_overlaps, gt_assignment, overlaps, SearchType::COLUMNWISE);

  // Sample foreground/background indices (select boxes with IOU within given limits).
  output.fg_inds = sampleIndices(max_overlaps, input.fg_params, input.rois_per_image);
  output.bg_inds = sampleIndices(max_overlaps, input.bg_params, input.rois_per_image - output.fg_inds.size());

  // Keep the selected indices.
  output.keep_inds.reserve(output.fg_inds.size() + output.bg_inds.size());
  output.keep_inds.insert(output.keep_inds.end(), output.fg_inds.begin(), output.fg_inds.end());
  output.keep_inds.insert(output.keep_inds.end(), output.bg_inds.begin(), output.bg_inds.end());

  // Select sampled values from various vectors.
  output.labels.reserve(output.keep_inds.size());
  for (auto const & i : output.keep_inds) { output.labels.push_back(gt_boxes.at<T>(gt_assignment[i], 4)); }

  // Clamp labels for the background ROIs to 0.
  std::fill(output.labels.begin() + output.fg_inds.size(), output.labels.end(), 0);

  // Prepare data for computing regression targets.
  std::vector<cv::Mat> gts;
  output.rois.reserve(output.keep_inds.size());
  gts.reserve(output.keep_inds.size());

  for (auto const & i : output.keep_inds) {
    output.rois.push_back(all_boxes.row(i));
    gts.push_back(gt_boxes.row(gt_assignment[i]));
  }

  // Compute bounding box regression targets (of external ROIs with respect to GT ROIs).
  cv::Mat target_data = algorithms::computeTargets<T>(output.rois, gts, input.sampling_mean,
                                                      input.sampling_std);
  // Create regression data (targets and loss weights) in the format desired by the network.
  algorithms::getRegressionLabels<T>(output.target_mat, output.in_weights, target_data,
                                     output.labels, input.num_classes, input.regression_weights);

  // Create outside regression weights.
  cv::threshold(output.in_weights, output.out_weights, 0, 1, CV_THRESH_BINARY);

  // If MNC mode is set to true, do the following.
  if (!input.mask_info.empty() && !input.gt_masks.empty()) {
    // Map to original image space.
    cv::Mat scaled_rois(output.rois.size(), 4, cv::DataType<T>::type);
    for (size_t i = 0; i < output.rois.size(); ++i) { 
      for (int j = 0; j < output.rois[i].cols; ++j) {
        scaled_rois.at<T>(i, j) = output.rois[i].at<T>(0, j) / input.image_scale;
      }
    }
    cv::Mat scaled_gt_boxes = gt_boxes * (1.0 / input.image_scale);

    // Allocate memory.
    output.pos_masks.reserve(output.keep_inds.size());
    output.mask_weight.reserve(output.rois.size());

    // Only assign box-level foreground as positive mask regression.
    for (size_t i = 0; i < output.rois.size(); ++i) {
      output.mask_weight.emplace_back(input.mask_size, input.mask_size, cv::DataType<T>::type,
                                      i < output.fg_inds.size() ? 1 : 0);
    }

    // Set top_mask_info to -1 and fill it with values only for the rows corresponding to fg_inds.
    output.top_mask_info = cv::Mat(output.keep_inds.size(), 12, cv::DataType<T>::type, -1);

    int i = 0;
    std::set<size_t>::iterator it;
    for (it = output.fg_inds.begin(); it != output.fg_inds.end(); it++, i++) {
      int j = gt_assignment[*it];
      cv::Size gt_mask_info(input.mask_info.at<T>(j, 1), input.mask_info.at<T>(j, 0));

      // Select the appropriate GT mask.
      cv::Range ranges[3] = {{j, j + 1}, {0, gt_mask_info.height}, {0, gt_mask_info.width}};
      cv::Mat gt_mask3(input.gt_masks, ranges);

      // Select the scaled external and ground-truth boxes.
      cv::Mat ex_box = scaled_rois.row(i).clone();
      cv::Mat gt_box = scaled_gt_boxes.row(j).clone();

      // Convert coordinates to integers.
      for (int k = 0; k < 4; ++k) {
        ex_box.at<T>(0, k) = std::round(ex_box.at<T>(0, k));
        gt_box.at<T>(0, k) = std::round(gt_box.at<T>(0, k));
      }

      // Calculate intersection between external and ground-truth boxes and mask it according to gt_mask.
      output.pos_masks.push_back(algorithms::intersectMask<T>(
        ex_box, gt_box, gt_mask3, input.mask_size, input.binarize_thresh));

      output.top_mask_info.at<T>(i,  0) = j;
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

    // Fill the pos_masks with zeros for the indices corresponding to backgrounds.
    while (output.pos_masks.size() != output.pos_masks.capacity()) {
      output.pos_masks.emplace_back(input.mask_size, input.mask_size, CV_8U, 0);
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
  (void) bottom;

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
  (void) bottom;
  (void) top;
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

  iteration_++;

  ROISamplingInput  input;
  ROISamplingOutput output;

  // Extract ROIs (x1, y1, x2, y2) and their labels (column 0 for all_rois and column 4 for gt_rois).
  input.all_rois  = blob::extract<Dtype>(*bottom[0], {0, 0}, bottom[0]->shape()).clone();
  input.gt_rois   = blob::extract<Dtype>(*bottom[1], {0, 0}, bottom[1]->shape());

  // Potentially dangerous, will likely modify data from bottom[0]!
  cv::Mat temp = cv::Mat::zeros(input.gt_rois.rows, 5, input.gt_rois.type());
  input.gt_rois(cv::Rect(0, 0, 4, input.gt_rois.rows)).copyTo(temp(cv::Rect(1, 0, 4, input.gt_rois.rows)));
  input.all_rois.push_back(temp);

  // Extract ground-truth masks and mask info.
  if (params_.mnc_mode()) {
    input.gt_masks  = blob::extract<Dtype>(*bottom[3], {0, 0, 0}, bottom[3]->shape());
    input.mask_info = blob::extract<Dtype>(*bottom[4], {0, 0},    bottom[4]->shape());
  }

  for (size_t i = 0; i < params_.fg_fraction_size(); ++i) {
    input.fg_params.emplace_back(params_.fg_fraction(i), params_.fg_lo_thresh(i), params_.fg_hi_thresh(i));
  }

  for (size_t i = 0; i < params_.bg_fraction_size(); ++i) {
    input.bg_params.emplace_back(params_.bg_fraction(i), params_.bg_lo_thresh(i), params_.bg_hi_thresh(i));
  }

  input.rois_per_image     = params_.batch_size();
  input.num_classes        = params_.num_classes();
  input.image_scale        = bottom[2]->data_at({0, 2});
  input.mask_size          = params_.mask_size();
  input.binarize_thresh    = params_.binarize_thresh();
  input.sampling_mean      = cv::Mat::zeros(1, 4, cv::DataType<Dtype>::type);
  input.sampling_std       = (cv::Mat_<Dtype>(1, 4) << 0.1, 0.1, 0.2, 0.2);
  input.regression_weights = cv::Mat::ones(1, 4, cv::DataType<Dtype>::type);

  sampleROIs<Dtype>(output, input);
  keep_indices_ = params_.bp_all() ? output.keep_inds : utils::convert<size_t, size_t>(output.fg_inds);
  processTopBlobs(top, bottom, keep_indices_, output, input, params_.mix_index(), iteration_);
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

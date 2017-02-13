#include "caffe/layers/proposal_target_layer.hpp"
#include "caffe/layers/proposal_common.hpp"
#include "caffe/layers/proposal_anchor_transforms.hpp"
#include "caffe/layers/proposal_mask_transforms.hpp"
#include "caffe/layers/proposal_rectangle_transforms.hpp"

#include <set>

namespace caffe {
namespace proposal_layer {

struct RoiParameters {
  float fraction;
  float lo_thresh;
  float hi_thresh;

  RoiParameters(float const f, float const lt, float const ht):
    fraction(f), lo_thresh(lt), hi_thresh(ht) {}
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
                                std::vector<RoiParameters> const & params,
                                int                        const   num_rois)
{
  std::set<size_t> indices;

  for (auto const & p : params) {
    std::vector<size_t> thresh_indices = thresholdOverlaps(overlaps, p.lo_thresh, p.hi_thresh);
    //size_t num_current_rois = std::min(thresh_indices.size(), static_cast<size_t>(std::round(num_rois * p.fraction)));

    //if (!thresh_indices.empty()) { thresh_indices = sampleWithoutReplacement(thresh_indices, num_current_rois); }

    indices.insert(thresh_indices.begin(), thresh_indices.end());
  }

  return indices;
}

template <typename T>
void sampleRois(std::vector<cv::Rect_<T>>        & rois,
                std::vector<int>                 & labels,
                cv::Mat_<T>                      & target_mat,
                cv::Mat_<T>                      & in_weights,
                cv::Mat_<T>                      & out_weights,
                cv::Mat_<T>                      & top_mask_info,
                std::vector<cv::Mat_<T>>         & mask_weight,
                std::vector<cv::Mat_<T>>         & pos_masks,

                std::vector<cv::Rect_<T>>  const & all_rois,
                std::vector<int>           const & all_labels,
                std::vector<cv::Rect_<T>>  const & gt_boxes,
                std::vector<int>           const & gt_labels,
                int                        const   rois_per_image,
                int                        const   num_classes,
                std::vector<cv::Mat_<T>>   const & gt_masks,
                float                      const   image_scale,
                cv::Mat_<T>                const & mask_info,

                std::set<size_t>                 & fg_inds,
                std::set<size_t>                 & bg_inds,
                std::vector<size_t>              & keep_inds,
                std::vector<RoiParameters> const & fg_params,
                std::vector<RoiParameters> const & bg_params)
{
  std::vector<int> gt_assignment;
  std::vector<T> max_overlaps;

  cv::Mat_<T> overlaps = rectangle::intersectionOverUnion(all_rois, gt_boxes);
  algorithms::max(max_overlaps, gt_assignment, overlaps, SearchType::COLUMNWISE);

  fg_inds = sampleIndices(max_overlaps, fg_params, rois_per_image);
  bg_inds = sampleIndices(max_overlaps, bg_params, rois_per_image - fg_inds.size());

  keep_inds.reserve(fg_inds.size() + bg_inds.size());
  keep_inds.insert(keep_inds.end(), fg_inds.begin(), fg_inds.end());
  keep_inds.insert(keep_inds.end(), bg_inds.begin(), bg_inds.end());

  labels = utils::select(utils::select(gt_labels, gt_assignment), keep_inds);
  std::fill(labels.begin() + fg_inds.size(), labels.end(), 0);

  rois                                  = utils::select(all_rois, keep_inds);
  std::vector<cv::Rect_<T>> gts         = utils::select(gt_boxes, utils::select(gt_assignment, keep_inds));
  std::vector<cv::Rect_<T>> target_data = algorithms::computeTargets(rois, gts);

  algorithms::getRegressionLabels(target_mat, in_weights, target_data, labels, num_classes);
  cv::threshold(in_weights, out_weights, 0, 1, CV_THRESH_BINARY);

  if (!mask_info.empty() && !gt_masks.empty()) {
    std::vector<cv::Rect_<T>> scaled_rois     = rectangle::scaledRectangles(rois, image_scale);
    std::vector<cv::Rect_<T>> scaled_gt_boxes = rectangle::scaledRectangles(gt_boxes, image_scale);

    int mask_size = 21;
    float threshold = 0.4;

    pos_masks.reserve(keep_inds.size());
    mask_weight.reserve(rois.size());

    for (size_t i = 0; i < fg_inds.size(); ++i)           { mask_weight.push_back(cv::Mat_<T>::ones(mask_size, mask_size)); }
    for (size_t i = fg_inds.size(); i < rois.size(); ++i) { mask_weight.push_back(cv::Mat_<T>::zeros(mask_size, mask_size)); }

    top_mask_info = cv::Mat_<T>::zeros(keep_inds.size(), 12);
    top_mask_info(cv::Range(fg_inds.size(), top_mask_info.rows), cv::Range(0, top_mask_info.cols)).setTo(-1);

    std::set<size_t>::iterator it;
    for (it = fg_inds.begin(); it != fg_inds.end(); ++it) {
      int i = std::distance(fg_inds.begin(), it);


      cv::Size gt_mask_info(mask_info.template at<T>(gt_assignment[*it], 1),
                            mask_info.template at<T>(gt_assignment[*it], 0));

      cv::Rect_<T> gt_box = scaled_gt_boxes[gt_assignment[*it]];
      cv::Rect_<T> ex_box = scaled_rois[i];
      cv::Mat_<T> gt_mask = gt_masks[gt_assignment[*it]](cv::Rect(0, 0, gt_mask_info.width, gt_mask_info.height));

      cv::Mat_<T> mask = algorithms::intersectMask(ex_box, gt_box, gt_mask, mask_size, threshold);
      pos_masks.push_back(mask);

      top_mask_info.template at<T>(i,  0) = gt_assignment[*it];
      top_mask_info.template at<T>(i,  1) = gt_mask_info.height;
      top_mask_info.template at<T>(i,  2) = gt_mask_info.width;
      top_mask_info.template at<T>(i,  3) = labels[i];
      top_mask_info.template at<T>(i,  4) = std::round(ex_box.tl().x);
      top_mask_info.template at<T>(i,  5) = std::round(ex_box.tl().y);
      top_mask_info.template at<T>(i,  6) = std::round(ex_box.br().x);
      top_mask_info.template at<T>(i,  7) = std::round(ex_box.br().y);
      top_mask_info.template at<T>(i,  8) = std::round(gt_box.tl().x);
      top_mask_info.template at<T>(i,  9) = std::round(gt_box.tl().y);
      top_mask_info.template at<T>(i, 10) = std::round(gt_box.br().x);
      top_mask_info.template at<T>(i, 11) = std::round(gt_box.br().y);
    }

    while (pos_masks.size() != pos_masks.capacity()) {
      pos_masks.emplace_back(cv::Mat_<T>::zeros(mask_size, mask_size));
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

  std::vector<cv::Rect_<Dtype>> all_rois   = blob::extractRectsFromMatrix<Dtype, Dtype>(*bottom[0], 1);
  std::vector<int>              all_labels = blob::extractVector<int, Dtype>(*bottom[0], {0, 0}, {bottom[0]->shape()[0], 1});
  std::vector<cv::Rect_<Dtype>> gt_boxes   = blob::extractRectsFromMatrix<Dtype, Dtype>(*bottom[1], 0);
  std::vector<int>              gt_labels  = blob::extractVector<int, Dtype>(*bottom[1], {0, 4}, {bottom[1]->shape()[0], 5});

  iteration_++;

  std::vector<cv::Mat_<Dtype>> gt_masks;
  cv::Mat_<Dtype> mask_info;

  if (params_.mnc_mode()) {
    gt_masks  = blob::extractVectorOfMatrices<Dtype, Dtype>(*bottom[3], {0, 0, 0}, bottom[3]->shape());
    mask_info = blob::extractMatrix<Dtype, Dtype>(*bottom[4], {0, 0}, bottom[4]->shape());
  }

  all_rois.insert(all_rois.end(), gt_boxes.begin(), gt_boxes.end());
  all_labels.insert(all_labels.end(), gt_boxes.size(), 0);

  std::vector<proposal_layer::RoiParameters> fg_params;
  std::vector<proposal_layer::RoiParameters> bg_params;

  for (size_t i = 0; i < params_.fg_fraction_size(); ++i) {
    fg_params.push_back({params_.fg_fraction(i), params_.fg_lo_thresh(i), params_.fg_hi_thresh(i)}); }
  for (size_t i = 0; i < params_.bg_fraction_size(); ++i) {
    bg_params.push_back({params_.bg_fraction(i), params_.bg_lo_thresh(i), params_.bg_hi_thresh(i)}); }

  std::vector<cv::Rect_<Dtype>> rois;
  std::vector<int> labels;
  cv::Mat_<Dtype> target_mat;
  cv::Mat_<Dtype> in_weights;
  cv::Mat_<Dtype> out_weights;
  cv::Mat_<Dtype> top_mask_info;
  std::vector<cv::Mat_<Dtype>> mask_weight;
  std::vector<cv::Mat_<Dtype>> pos_masks;
  std::set<size_t> fg_inds;
  std::set<size_t> bg_inds;
  std::vector<size_t> keep_inds;

  proposal_layer::sampleRois(rois,                 labels,                target_mat,  in_weights,
                             out_weights,          top_mask_info,         mask_weight, pos_masks,
                             all_rois,             all_labels,            gt_boxes,    gt_labels,
                             params_.batch_size(), params_.num_classes(), gt_masks,    bottom[2]->data_at({0, 2}),
                             mask_info,            fg_inds,               bg_inds,     keep_inds,
                             fg_params,            bg_params);

  keep_indices_ = params_.bp_all() ? keep_inds : utils::convert<size_t, size_t>(fg_inds);

  if (iteration_ == 40) {
    //exit(0);
  }


  //std::cerr << "--------------------------------------------------------" << std::endl;
  //std::cerr << std::endl << "top 0: " << std::endl;

  top[0]->Reshape({int(keep_inds.size()), 5});
  Dtype * top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < rois.size(); ++i) {
    top_data[i * 5]     = all_labels[keep_inds[i]];
    top_data[i * 5 + 1] = rois[i].tl().x;
    top_data[i * 5 + 2] = rois[i].tl().y;
    top_data[i * 5 + 3] = rois[i].br().x;
    top_data[i * 5 + 4] = rois[i].br().y;
    //std::cerr << std::setprecision(5) << top_data[i*5]   << " " << top_data[i*5+1] << " " <<
                                         //top_data[i*5+2] << " " << top_data[i*5+3] << " " <<
                                         //top_data[i*5+4] << std::endl;
  }

  blob::writeVector(*top[1], labels);
  //std::cerr << std::endl << "top 1: " << std::endl;
  //utils::print(labels);

  blob::writeMatrix(*top[2], target_mat);
  //std::cerr << std::endl << "top 2: " << std::endl;
  //std::cerr << std::setprecision(5) << target_mat << std::endl;

  blob::writeMatrix(*top[3], in_weights);
  //std::cerr << std::endl << "top 3: " << std::endl;
  //std::cerr << std::setprecision(5) << in_weights << std::endl;

  blob::writeMatrix(*top[4], out_weights);
  //std::cerr << std::endl << "top 4: " << std::endl;
  //std::cerr << std::setprecision(5) << out_weights << std::endl;

  if (!pos_masks.empty()) {
    top[5]->Reshape({int(pos_masks.size()), 1, pos_masks[0].rows, pos_masks[0].cols});
  } else {
    top[5]->Reshape({0, 1, 0, 0});
  }

  //std::cerr << std::endl << "top 5: " << std::endl;
  top_data = top[5]->mutable_cpu_data();
  for (size_t i = 0; i < pos_masks.size(); ++i) {
    for (int j = 0; j < pos_masks[0].rows; ++j) {
      for (int k = 0; k < pos_masks[0].cols; ++k) {
        int offset = top[5]->offset({int(i), 0, j, k});
        top_data[offset] = pos_masks[i].template at<Dtype>(j, k) / 255;
        //std::cerr << std::setprecision(5) << top_data[offset] << " ";
      }
      //std::cerr << std::endl;
    }
  }

  if (!pos_masks.empty()) {
    top[5]->Reshape({int(mask_weight.size()), 1, mask_weight[0].rows, mask_weight[0].cols});
  } else {
    top[5]->Reshape({0, 1, 0, 0});
  }

  //std::cerr << std::endl << "top 6: " << std::endl;
  top[6]->Reshape({int(mask_weight.size()), 1, mask_weight[0].rows, mask_weight[0].cols});
  top_data = top[6]->mutable_cpu_data();

  for (size_t i = 0; i < mask_weight.size(); ++i) {
    for (int j = 0; j < mask_weight[0].rows; ++j) {
      for (int k = 0; k < mask_weight[0].cols; ++k) {
        int offset = top[6]->offset({int(i), 0, j, k});
        top_data[offset] = mask_weight[i].template at<Dtype>(j, k);
        //std::cerr << std::setprecision(5) << top_data[offset] << " ";
      }
      //std::cerr << std::endl;
    }
  }

  blob::writeMatrix(*top[7], top_mask_info);
  //std::cerr << std::endl << "top 7: " << std::endl;
  //std::cerr << top_mask_info << std::endl;

  if (params_.mix_index()) {
    std::vector<int> all_rois_index = blob::extractVector<int, Dtype>(*bottom[5], {0, 0}, {1, bottom[5]->shape()[1]});
    std::set<size_t> lower          = utils::compareLower(fg_inds, all_rois_index.size());

    auto x = utils::select(all_rois_index, lower);
    auto y = utils::select(all_rois_index, bg_inds);

    //std::cerr << std::endl << "top 8: " << std::endl;
    //utils::print(x);
    //std::cerr << std::endl << "top 9: " << std::endl;
    //utils::print(y);

    blob::writeVector(*top[8], x);
    blob::writeVector(*top[9], y);
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

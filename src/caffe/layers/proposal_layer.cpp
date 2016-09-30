#include <cfloat>

#include "caffe/layers/proposal_layer.hpp"

namespace caffe {

using Rectangle = proposal_layer::Rectangle;
using Point     = proposal_layer::Point;

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(std::vector<Blob<Dtype>*> const & bottom,
                                      std::vector<Blob<Dtype>*> const & top)
{
  ProposalParameter const proposal_param = this->layer_param_.proposal_param();

  feat_stride_      = proposal_param.feat_stride();
  clip_denominator_ = proposal_param.clip_base();
  clip_threshold_   = 1.0 / clip_denominator_;
  use_clip_         = proposal_param.use_clip();

  std::vector<float> ratios = { 0.5, 1, 2 };
  std::vector<float> scales = { 8, 16, 32 };

  generateReferenceAnchors(reference_anchors_, ratios, scales, 16);
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(std::vector<Blob<Dtype>*> const & bottom,
                                   std::vector<Blob<Dtype>*> const & top)
{
  top[0]->Reshape({ 1, 5 });
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(std::vector<Blob<Dtype>*> const & bottom,
                                       std::vector<Blob<Dtype>*> const & top)
{
  assert(bottom[0]->shape()[0] == 1);

  int layer_height = bottom[0]->shape()[2];
  int layer_width  = bottom[0]->shape()[3];

  int image_height = bottom[2]->data_at({0, 0});
  int image_width  = bottom[2]->data_at({0, 1});

  int pre_nms_top_n  = 6000;
  int post_nms_top_n = 300;
  float nms_thresh   = 0.7;
  int min_size       = 16;

  std::vector<float> scores = generateScoresVector(*bottom[0], reference_anchors_.size());
  std::cout << "marker" << std::endl;
  proposal_layer::printVector<float>(scores);

  std::vector<Rectangle> shifted_anchors;
  generateShiftedAnchors(shifted_anchors, reference_anchors_, layer_width, layer_height, feat_stride_);
  clipRectanglesToBounds(anchor_indexes_, shifted_anchors, image_width, image_height, false);

  std::vector<Rectangle> proposals;
  generateProposals(proposals, shifted_anchors, bottom[1]);
  clipRectanglesToBounds(proposal_indexes_, proposals, image_width, image_height, true);

  filter_indexes_ = getLargeRectangles(proposals, 32);
  std::vector<Rectangle> filtered_proposals;

  std::transform(filter_indexes_.begin(), filter_indexes_.end(),
                 std::back_inserter(filtered_proposals),
                 [proposals](int i) { return proposals[i]; });
}

/** \brief Applies delta transformation on given anchors, to get transformed proposals.
 *  \param [out] proposals Vector of proposals.
 *  \param [in]  anchors   Vector of anchors.
 *  \param [in]  deltas    Blob with delta transformations.
 *  \return True if proposals are returned, false otherwise.
 */
template <typename Dtype>
bool ProposalLayer<Dtype>::generateProposals(std::vector<Rectangle>          & proposals,
                                             std::vector<Rectangle>    const & anchors,
                                             Blob<Dtype>               const * deltas)
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
std::vector<int> ProposalLayer<Dtype>::getLargeRectangles(std::vector<Rectangle> const & rectangles,
                                                          float                  const   min_size)
{
  std::vector<int> indexes;
  
  for (size_t i = 0; i < rectangles.size(); ++i) {
    if (rectangles[i].width >= min_size && rectangles[i].height >= min_size) {
      indexes.push_back(i);
    }
  }

  return indexes;
}

template <typename Dtype>
void ProposalLayer<Dtype>::clipRectanglesToBounds(std::vector<int>             & indexes,
                                                  std::vector<Rectangle>       & rectangles,
                                                  int                    const   width,
                                                  int                    const   height,
                                                  bool                   const   auto_clip)
{
  indexes.reserve(rectangles.size());

  for (size_t i = 0; i < rectangles.size(); ++i) {
    Point top_left     = rectangles[i].topLeft();
    Point bottom_right = rectangles[i].bottomRight();

    if (top_left.x >= 0 && bottom_right.x < width &&
        top_left.y >= 0 && bottom_right.y < height) { indexes.push_back(i); }

    if (auto_clip) {
      top_left.x     = std::max(0.f, std::min(top_left.x,     static_cast<float>(width  - 1)));
      top_left.y     = std::max(0.f, std::min(top_left.y,     static_cast<float>(height - 1)));
      bottom_right.x = std::max(0.f, std::min(bottom_right.x, static_cast<float>(width  - 1)));
      bottom_right.y = std::max(0.f, std::min(bottom_right.y, static_cast<float>(height - 1)));

      rectangles[i] = Rectangle::fromCoordinates(top_left, bottom_right);
    }
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(std::vector<Blob<Dtype>*> const & top,
                                        std::vector<bool>         const & propagate_down,
                                        std::vector<Blob<Dtype>*> const & bottom)
{
  NOT_IMPLEMENTED;
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
std::vector<float> ProposalLayer<Dtype>::generateScoresVector(Blob<Dtype> const & blob, int const offset) {
  std::vector<float> scores;
  std::vector<int> shape = blob.shape();

  scores.resize(shape[2] * shape[3] * shape[1]);

  int index = 0;
  for (int j = 0; j < shape[2]; ++j) {
    for (int k = 0; k < shape[3]; ++k) {
      for (int i = offset; i < shape[1]; ++i) { scores[index++] = blob.data_at(0, i, j, k); }
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

#ifdef CPU_ONLY
STUB_GPU(ProposalLayer);
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}

#pragma once

#include "caffe/blob.hpp"
#include "caffe/util/rng.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.inl.hpp>

#include <algorithm>
#include <numeric>

namespace caffe {
namespace proposal_layer {

enum SearchType { COLUMNWISE, ROWWISE };

namespace blob {
  /** \brief Extracts a vector of rectangles from a blob.
   *  \param [in]  blob        Input blob.
   *  \param [in]  col_index   Index of the column with the X coordinates.
   *  \return Vector of rectangles containing data from the blob
   */
  template <typename A, typename B>
  std::vector<cv::Rect_<A>> extractRectsFromMatrix(Blob<B> const & layer, int const col_index) {
    assert(layer.shape().size() == 2 && col_index >= 0 && col_index + 3 < layer.shape()[1]);

    std::vector<cv::Rect_<A>> rectangles;
    rectangles.reserve(layer.shape()[0]);

    for (int i = 0; i < layer.shape()[0]; ++i) {
      A tl_x = layer.data_at({i, col_index});
      A tl_y = layer.data_at({i, col_index + 1});
      A br_x = layer.data_at({i, col_index + 2});
      A br_y = layer.data_at({i, col_index + 3});

      rectangles.emplace_back(cv::Point_<A>{tl_x, tl_y}, cv::Point_<A>{br_x, br_y});
    }

    return rectangles;
  }

  /** \brief Extracts a vector from a Caffe blob.
   *  \param [in]  blob    Input Caffe blob.
   *  \param [in]  start   Vector of start indices.
   *  \param [in]  end     Vector of end indices.
   *  \return Vector of elements from the blob.
   */
  template <typename A, typename B>
  std::vector<A> extractVector(Blob<B>          const & blob,
                               std::vector<int> const & start,
                               std::vector<int> const & end)
  {
    assert(start.size() >= 1 && start.size() == end.size() == blob.shape().size());

    bool found_indices = false;
    int index          = 0;

    for (int i = 0; i < int(start.size()); ++i) {
      assert(start[i] >= 0 && end[i] <= blob.shape()[i] && start[i] < end[i]);

      if (found_indices) { assert(start[i] + 1 == end[i]); continue; }
      if (start[i] + 1 == end[i]) { continue; }

      found_indices = true; index = i;
    }

    assert(found_indices);

    std::vector<A> result;
    result.reserve(end[index] - start[index]);

    std::vector<int> indices = start;
    for (int i = start[index]; i < end[index]; ++i) {
      indices[index] = i;
      result.push_back(A(blob.data_at(indices)));
    }

    return result;
  }

  template <typename A, typename B>
  cv::Mat_<A> extractMatrix(Blob<B>          const & blob,
                            std::vector<int> const & start,
                            std::vector<int> const & end)
  {
    assert(start.size() >= 2 && start.size() == end.size() == blob.shape().size());

    bool found_row_indices = false;
    bool found_col_indices = false;

    int row_index = 0;
    int col_index = 0;

    for (int i = 0; i < static_cast<int>(start.size()); ++i) {
      assert(start[i] >= 0 && end[i] <= blob.shape()[i] && start[i] < end[i]);

      if (found_row_indices && found_col_indices) { assert(start[i] + 1 == end[i]); continue;  }
      if (start[i] + 1 == end[i])                 { continue;  }

      if (!found_row_indices) { found_row_indices = true; row_index = i; continue;  }
      found_col_indices = true; col_index = i;
    }

    assert(found_row_indices && found_col_indices);

    cv::Mat_<A> result(end[row_index] - start[row_index], end[col_index] - start[col_index]);
    std::vector<int> indices = start;
    for (int i = start[row_index]; i < end[row_index]; ++i) {
      for (int j = start[col_index]; j < end[col_index]; ++j) {
        indices[row_index] = i;
        indices[col_index] = j;
        result.template at<A>(i - start[row_index], j - start[col_index]) = blob.data_at(indices);
      }
    }

    return result;
  }


  /** \brief Extracts an N-dimensional matrix from a Caffe blob.
   *  \param [in]  blob    Input Caffe blob.
   *  \param [in]  start   Vector of start indices.
   *  \param [in]  end     Vector of end indices.
   *  \return Matrix with the elements from the blob.
   */
  template <typename T>
  cv::Mat extract(Blob<T>          const & blob,
                  std::vector<int> const & start,
                  std::vector<int> const & end)
  {
    CHECK(start.size() >= 2 && start.size() <= 4);
    CHECK(start.size() == end.size() && start.size() == blob.shape().size());
    cv::Mat data(blob.shape().size(), &blob.shape()[0], cv::DataType<T>::type, const_cast<T*>(blob.cpu_data()));

    cv::Range ranges[3];
    for (size_t i = 0; i < start.size(); ++i) {
      CHECK(start[i] >= 0 && end[i] <= blob.shape(i) && start[i] < end[i]);
      ranges[i] = cv::Range(start[i], end[i]);
    }

    return data(ranges);
  }

  /** \brief Extracts a vector of matrices from a Caffe blob.
   *  \param [in]  blob    Input Caffe blob.
   *  \param [in]  start   Vector of start indices.
   *  \param [in]  end     Vector of end indices.
   *  \return Vector of matrices containing elements from the blob.
   */
  //template <typename T>
  //std::vector<cv::Mat> extract(Blob<T>          const & blob,
                               //std::vector<int> const & start,
                               //std::vector<int> const & end)
  //{
    //CHECK_GE(start.size(), 2);
    //CHECK(start.size() == end.size() && start.size() == blob.shape().size());

    //cv::Mat data(blob.shape().size(), &blob.shape()[0], cv::DataType<T>::type,
                 //const_cast<T*>(blob.cpu_data()));

    //std::vector<cv::Range> ranges;
    //for (size_t i = 0; i < start.size(); ++i) {
      //CHECK(start[i] >= 0 && end[i] <= blob.shape(i) && start[i] < end[i]);
      //ranges.emplace_back(start[i], end[i]);
    //}

    //return cv::Mat(data, &ranges[0]);
  //}

  /** \brief Extracts a vector of matrices from a Caffe blob.
   *  \param [in]  blob    Input Caffe blob.
   *  \param [in]  start   Vector of start indices.
   *  \param [in]  end     Vector of end indices.
   *  \return Vector of matrices containing elements from the blob.
   */
  template <typename A, typename B>
  std::vector<cv::Mat_<A>> extractVectorOfMatrices(Blob<B>          const & blob,
                                                   std::vector<int> const & start,
                                                   std::vector<int> const & end)
  {
    assert(start.size() >= 3 && start.size() == end.size() == blob.shape().size());

    bool found_slice_indices = false;
    bool found_row_indices   = false;
    bool found_col_indices   = false;

    int slice_index = 0;
    int row_index   = 0;
    int col_index   = 0;

    for (int i = 0; i < int(start.size()); ++i) {
      assert(start[i] >= 0 && end[i] <= blob.shape()[i] && start[i] < end[i]);

      if (found_slice_indices && found_row_indices && found_col_indices) { assert(start[i] + 1 == end[i]); continue; }

      if (!found_slice_indices) { found_slice_indices = true; slice_index = i; continue; }
      if (!found_row_indices)   { found_row_indices   = true; row_index   = i; continue; }

      found_col_indices = true; col_index = i;
    }

    assert(found_slice_indices && found_row_indices && found_col_indices);

    std::vector<cv::Mat_<A>> result;
    result.reserve(end[slice_index] - start[slice_index]);

    std::vector<int> indices = start;
    for (int i = start[slice_index]; i < end[slice_index]; ++i) {
      cv::Mat_<A> matrix(end[row_index] - start[row_index], end[col_index] - start[col_index]);
      for (int j = start[row_index]; j < end[row_index]; ++j) {
        for (int k = start[col_index]; k < end[col_index]; ++k) {
          indices[slice_index] = i;
          indices[row_index]   = j;
          indices[col_index]   = k;

          matrix.template at<A>(j - start[row_index], k - start[col_index]) = blob.data_at(indices);
        }
      }
      result.push_back(matrix);
    }

    return result;
  }

  /** \brief Writes the information contained in an OpenCV matrix to a blob.
   *  \param [out]  blob     Caffe blob.
   *  \param [in]   matrix   OpenCV matrix.
   */
  template <typename T>
  void writeMatrix(Blob<T> & blob, cv::Mat const & matrix) {
    blob.Reshape({matrix.rows, matrix.cols});
    T * data = blob.mutable_cpu_data();
    for (size_t i = 0; i < matrix.total(); ++i) { data[i] = matrix.at<T>(i); }
  }


  /** \brief Writes the information contained in an OpenCV matrix to a blob.
   *  \param [out]  blob     Caffe blob.
   *  \param [in]   matrix   OpenCV matrix.
   */
  template <typename A, typename B>
  void writeMatrix(Blob<A> & blob, cv::Mat_<B> const & matrix) {
    blob.Reshape({matrix.rows, matrix.cols});
    A * data = blob.mutable_cpu_data();
    for (int i = 0; i < matrix.total(); ++i) { data[i] = matrix.template at<B>(i); }
  }

  /** \brief Writes the information contained in a vector to a blob.
   *  \param [out]  blob   Caffe blob.
   *  \param [in]   vec    Data vector.
   */
  template <typename A, typename B>
  void writeVector(Blob<A> & blob, std::vector<B> const & vec) {
    blob.Reshape({int(vec.size())});
    A * data = blob.mutable_cpu_data();
    for (size_t i = 0; i < vec.size(); ++i) { data[i] = vec[i]; }
  }

  /** \brief Writes the information contained in a set to a blob.
   *  \param [out]  blob   Caffe blob.
   *  \param [in]   dset   Data set.
   */
  template <typename A, typename B>
  void writeSet(Blob<A> & blob, std::set<B> const & dset) {
    blob.Reshape({int(dset.size())});
    A * data = blob.mutable_cpu_data();

    int i = 0;
    for (auto const & elem : dset) { data[i++] = elem; }
  }
}

namespace rectangle {
  /** \brief Scales a rectangle by a given factor.
   *  \param [in]   r       Input rectangle.
   *  \param [in]   scale   Scaling factor.
   *  \return Scaled rectangle.
   */
  template <typename T>
  cv::Rect_<T> scaledRectangle(cv::Rect_<T> const & r, float const scale) {
    return { r.x / scale, r.y / scale, r.width / scale, r.height / scale };
  }

  /** \brief Scales a vector of rectangles by a given factor.
   *  \param [in]   rects   Vector of rectangles.
   *  \param [in]   scale   Scaling factor.
   *  \return Vector of scaled rectangles.
   */
  template <typename T>
  std::vector<cv::Rect_<T>> scaledRectangles(std::vector<cv::Rect_<T>> const & rects, float const scale) {
    std::vector<cv::Rect_<T>> scaled;
    scaled.reserve(rects.size());
    for (auto const & r : rects) { scaled.emplace_back(r.x / scale, r.y / scale, r.width / scale, r.height / scale); }
    return scaled;
  }

  /** \brief Creates a rectangle, given its center and dimensions.
   *  \param [in]   x        X coordinate of the center.
   *  \param [in]   y        Y coordinate of the center.
   *  \param [in]   width    Width of the rectangle.
   *  \param [in]   height   Height of the rectangle.
   *  \return Rectangle with the given center and dimensions.
   */
  template <typename T>
  cv::Rect_<T> centeredRectangle(T const x, T const y, T const width, T const height) {
    return { x - width / 2, y - height / 2, width, height };
  }

  /** \brief Computes the center of a given rectangle.
   *  \param [in]  rectangle   Input rectangle.
   *  \return Rectangle center.
   */
  template <typename T>
  cv::Point_<T> getRectangleCenter(cv::Rect_<T> const & rectangle) {
    return { rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2 };
  }

  /** \brief Computes intersection over union for two given sets of rectangles.
   *  \param  r   Input rectangles.
   *  \param  q   Input (query) rectangles.
   *  \return Intersection over union for the given pairs of rectangles.
   */
  template <typename T>
  cv::Mat_<T> intersectionOverUnion(std::vector<cv::Rect_<T>> const & r,
                                    std::vector<cv::Rect_<T>> const & q)
  {
    cv::Mat_<T> overlaps = cv::Mat_<T>::zeros(r.size(), q.size());

    for (size_t k = 0; k < q.size(); ++k) {
      T box_area = (q[k].br().x - q[k].tl().x + 1) * (q[k].br().y - q[k].tl().y + 1);
      for (size_t n = 0; n < r.size(); ++n) {
        T iw = std::min(r[n].br().x, q[k].br().x) - std::max(r[n].tl().x, q[k].tl().x) + 1;

        if (iw > 0) {
          T ih = std::min(r[n].br().y, q[k].br().y) - std::max(r[n].tl().y, q[k].tl().y) + 1;

          if (ih > 0) {
            T ua = (r[n].br().x - r[n].tl().x + 1) * (r[n].br().y - r[n].tl().y + 1) + box_area - iw * ih;
            overlaps.template at<T>(n, k) = iw * ih / ua;
          }
        }

        //cv::Rect_<T> overlap = q[k] & r[n];
        //overlaps.template at<T>(n, k) = overlap.area() / (r[n].area() + q[k].area() - overlap.area());
      }
    }

    return overlaps;
  }

  /** \brief Computes intersection over union for two given sets of rectangles.
   *  \param  r   Input rectangles.
   *  \param  q   Input (query) rectangles.
   *  \return Intersection over union for the given pairs of rectangles.
   */
  template <typename T>
  cv::Mat intersectionOverUnion(cv::Mat const & r, cv::Mat const & q) {
    CHECK_EQ(r.type(), q.type());
    cv::Mat overlaps = cv::Mat::zeros(r.rows, q.rows, r.type());

    for (int k = 0; k < q.rows; ++k) {
      T box_area = (q.at<T>(k, 2) - q.at<T>(k, 0) + 1) * (q.at<T>(k, 3) - q.at<T>(k, 1) + 1);
      for (int n = 0; n < r.rows; ++n) {
        T iw = std::min(r.at<T>(n, 2), q.at<T>(k, 2)) - std::max(r.at<T>(n, 0), q.at<T>(k, 0)) + 1;

        if (iw > 0) {
          T ih = std::min(r.at<T>(n, 3), q.at<T>(k, 3)) - std::max(r.at<T>(n, 1), q.at<T>(k, 1)) + 1;

          if (ih > 0) {
            T ua = (r.at<T>(n, 2) - r.at<T>(n, 0) + 1) * (r.at<T>(n, 3) - r.at<T>(n, 1) + 1) + box_area - iw * ih;
            overlaps.at<T>(n, k) = iw * ih / ua;
          }
        }

        //cv::Rect_<T> overlap = q[k] & r[n];
        //overlaps.template at<T>(n, k) = overlap.area() / (r[n].area() + q[k].area() - overlap.area());
      }
    }

    return overlaps;
  }


  /** \brief Filters out rectangles with width or height lower than a given threshold.
   *  \param [in]  rectangles   Vector of rectangles.
   *  \param [in]  min_size     Threshold to be used for filtering.
   *  \return Vector of indices corresponding to the filtered rectangles.
   */
  template <typename T>
  std::vector<size_t> getLargeRectangles(std::vector<cv::Rect_<T>> const & rectangles,
                                         float                     const   min_size)
  {
    std::vector<size_t> indices;
    for (size_t i = 0; i < rectangles.size(); ++i) {
      if (rectangles[i].width >= min_size && rectangles[i].height >= min_size) { indices.push_back(i); }
    }

    return indices;
  }

  /** \brief Clip rectangle dimensions to given boundaries.
   *  \param [in,out]  rectangles   Vector of rectangles.
   *  \param [in]      dimensions   Input dimensions.
   *  \param [in]      auto_clip    True if the dimensions of the rectangles will be changed, false otherwise.
   *  \return Vector of indices corresponding to the clipped rectangles.
   */
  template <typename T>
  std::vector<size_t> clipRectangles(std::vector<cv::Rect_<T>>       & rectangles,
                                     cv::Size                  const   dimensions,
                                     bool                      const   auto_clip = false)
  {
    std::vector<size_t> indices;

    for (size_t i = 0; i < rectangles.size(); ++i) {
      cv::Point_<T> const & tl = rectangles[i].tl();
      cv::Point_<T> const & br = rectangles[i].br();

      if (tl.x >= 0 && br.x <= dimensions.width && tl.y >= 0 && br.y <= dimensions.height) {
        indices.push_back(i);
      }

      if (auto_clip) {
        rectangles[i] &= cv::Rect_<T>(0, 0, T(dimensions.width), T(dimensions.height));
      }
    }

    return indices;
  }
}

namespace utils {
  /** \brief Convert set to vector.
   *  \param [in]  s   Input set.
   *  \return Vector with elements from the set.
   */
  template <typename T>
  std::set<T> compareLower(std::set<T> const & dset, T const & value) {
    std::set<T> result;
    for (auto const & elem : dset) { if (elem < value) { result.insert(elem); } }
    return result;
  }

  /** \brief Convert set to vector.
   *  \param [in]  s   Input set.
   *  \return Vector with elements from the set.
   */
  template <typename A, typename B>
  std::vector<A> convert(std::set<B> const & s) { return std::vector<A>(s.begin(), s.end()); }

  /** \brief Convert vector to set.
   *  \param [in]  v   Input vector.
   *  \return Set with elements from the vector.
   */
  template <typename A, typename B>
  std::set<A> convert(std::vector<B> const & v) { return std::set<A>(v.begin(), v.end()); }

  /** \brief Outputs the elements of a given vector.
   *  \param [in]   vector      Input vector.
   */
  template <typename T>
  void print(std::vector<T> const & vector) {
    for (auto const & element : vector) { std::cout << element << " "; }; std::cout << std::endl;
  }

  /** \brief Outputs the elements of a given vector.
   *  \param [in]   vector      Input vector.
   */
  template <typename T>
  void print(std::set<T> const & set) {
    for (auto const & element : set) { std::cout << element << " "; }; std::cout << std::endl;
  }

  /** \brief Selects elements from a vector, based on a given vector of indices.
   *  \param [in]   vector      Input vector.
   *  \param [in]   indices     Vector of indices.
   *  \return Vector with the selected elements.
   */
  template <typename A, typename B>
  std::vector<A> select(std::vector<A> const & vector, std::vector<B> const & indices) {
    std::vector<A> result;

    result.reserve(indices.size());
    for (auto const & i : indices) { result.push_back(vector[i]); }

    return result;
  }

  /** \brief Selects elements from a vector, based on a given set of indices.
   *  \param [in]   vector      Input vector.
   *  \param [in]   indices     Set of indices.
   *  \return Vector with the selected elements.
   */
  template <typename A, typename B>
  std::vector<A> select(std::vector<A> const & vector, std::set<B> const & indices) {
    std::vector<A> result;

    result.reserve(indices.size());
    for (auto const & i : indices) { result.push_back(vector[i]); }

    return result;
  }

  /** \brief Sorts a given vector in decreasing order.
   *  \param [in]   vector      Input vector.
   *  \return Indices of the sorted vector elements.
   */
  template <typename T>
  std::vector<size_t> sort(std::vector<T> const & vector) {
    std::vector<size_t> indices(vector.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&vector](size_t i, size_t j) { return vector[i] >= vector[j]; });
    return indices;
  }

  /** \brief Unmap a subset of item (data) back to the original set of items (of size count).
   *  \param [in] data  Input data.
   *  \param [in] count Size of output data.
   *  \param [in] inds  Vector of indices in data to copy.
   *  \param [in] fill  Value to use to fill the empty spots.
   *  \return Vector where entries from data is copied from based on indices in inds.
   */
  template <typename T>
  std::vector<T> unmap(std::vector<T> const & data, size_t count, std::vector<size_t> const & inds, T fill) {
    assert(inds.size() <= count);

    std::vector<T> ret(count, fill);
    for (size_t i = 0; i < inds.size(); ++i) {
      ret[inds[i]] = data[i];
    }

    return ret;
  }
}

namespace algorithms {
  /** \brief Implements the argmax function on a matrix.
   *  \param [in]   source        Input matrix.
   *  \param [in]   search_type   Search type.
   *  \return Vector of indices.
   */
  template <typename T>
  std::vector<int> argmax(cv::Mat_<T> const & source, SearchType const & search_type) {
    std::vector<int> indices;
    indices.reserve(search_type == SearchType::COLUMNWISE ? source.rows : source.cols);

    cv::Point location;
    for (int i = 0; i < int(indices.capacity()); ++i) {
      if (search_type == SearchType::COLUMNWISE) {
        cv::minMaxLoc(source.row(i), 0, 0, 0, &location);
      } else {
        cv::minMaxLoc(source.col(i), 0, 0, 0, &location);
      }

      indices.push_back(search_type == SearchType::COLUMNWISE ? location.x : location.y);
    }

    return indices;
  }

  /** \brief Implements the max function on a matrix.
   *  \param [out]  values        Vector with the max values.
   *  \param [out]  indices       Vector with the indices of the max values.
   *  \param [in]   source        Input matrix.
   *  \param [in]   search_type   Search type.
   *  \return Vector of max values.
   */
  template <typename T>
  void max(std::vector<T> & values, std::vector<int> & indices, cv::Mat_<T> const & source, SearchType const & search_type) {
    indices = argmax(source, search_type);
    values.reserve(indices.size());

    for (int i = 0; i < int(indices.size()); ++i) {
      values.push_back(search_type == SearchType::COLUMNWISE ? source.template at<T>(i, indices[i]) :
                                                               source.template at<T>(indices[i], i));
    }
  }

  /** \brief Implements the max function on a matrix.
   *  \param [in]   source        Input matrix.
   *  \param [in]   search_type   Search type.
   *  \return Vector of max values.
   */
  template <typename T>
  std::vector<T> max(cv::Mat_<T> const & source, SearchType const & search_type) {
    std::vector<int> indices;
    std::vector<T>   values;

    max(values, indices, source, search_type);

    return values;
  }

  /** \brief Sample elements from a vector, without replacement.
   *  \param [in]  vector        Input vector.
   *  \param [in]  num_samples   Number of samples.
   *  \return Vector with sampled elements.
   */
  template <typename T>
  std::vector<T> sampleWithoutReplacement(std::vector<T> const & vector, size_t const num_samples) {
    assert(num_samples <= vector.size());

    std::vector<size_t> indices(vector.size());
    std::iota(indices.begin(), indices.end(), 0);

    caffe::shuffle(indices.begin(), indices.begin() + num_samples);

    std::vector<T> samples = utils::select(vector, indices);

    return samples;
  }
}


}
}

#ifndef _ROS_NCNN_UTILS_HEADER_
#define _ROS_NCNN_UTILS_HEADER_

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


template <class T>
  inline float intersection_area(const T& a, const T& b) {
      cv::Rect_<float> inter = a.rect & b.rect;
      return inter.area();
  }

template <class T>
    void qsort_descent_inplace(std::vector<T>& objects, int left, int right) {
      int i = left;
      int j = right;
      float p = objects[(left + right) / 2].prob;

      while (i <= j)
      {
          while (objects[i].prob > p)
              i++;

          while (objects[j].prob < p)
              j--;

          if (i <= j)
          {
              // swap
              std::swap(objects[i], objects[j]);

              i++;
              j--;
          }
      }

      #pragma omp parallel sections
      {
          #pragma omp section
          {
              if (left < j) qsort_descent_inplace(objects, left, j);
          }
          #pragma omp section
          {
              if (i < right) qsort_descent_inplace(objects, i, right);
          }
      }
  }

template <class T>
  static void qsort_descent_inplace(std::vector<T>& objects)
  {
      if (objects.empty())
          return;

      qsort_descent_inplace(objects, 0, objects.size() - 1);
  }


template <class T>
  static void nms_sorted_bboxes(const std::vector<T>& objects, std::vector<int>& picked, float nms_threshold)
  {
      picked.clear();

      const int n = objects.size();

      std::vector<float> areas(n);
      for (int i = 0; i < n; i++)
      {
          areas[i] = objects[i].rect.area();
      }

      for (int i = 0; i < n; i++)
      {
          const T& a = objects[i];

          int keep = 1;
          for (int j = 0; j < (int)picked.size(); j++)
          {
              const T& b = objects[picked[j]];

              // intersection over union
              float inter_area = intersection_area(a, b);
              float union_area = areas[i] + areas[picked[j]] - inter_area;
              //             float IoU = inter_area / union_area
              if (inter_area / union_area > nms_threshold)
                  keep = 0;
          }

          if (keep)
              picked.push_back(i);
      }
  }


#endif

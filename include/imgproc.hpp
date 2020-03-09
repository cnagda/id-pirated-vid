#ifndef IMGPROC_HPP
#define IMGPROC_HPP

#include <opencv2/imgproc.hpp>

template<typename Matrix>
std::decay_t<Matrix> scaleToTarget(Matrix&& image, int targetWidth, int targetHeight){
    int srcWidth = image.cols;
    int srcHeight = image.rows;

    double ratio = std::min((double)targetHeight/srcHeight, (double)targetWidth/srcWidth);

    std::decay_t<Matrix> retval;

    cv::resize(image, retval, cv::Size(), ratio, ratio);
    return retval;
}

#endif
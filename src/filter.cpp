#include "filter.hpp"
#include "imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "matcher.hpp"
#include <cassert>
#include <iostream>
#include "vocabulary.hpp"

ordered_umat& ScaleImage::operator()(ordered_umat& image) const
{
    image.data = scaleToTarget(image.data, cropsize.first, cropsize.second);
    return image;
}
cv::UMat& ScaleImage::operator()(cv::UMat& image) const
{
    return image = scaleToTarget(image, cropsize.first, cropsize.second);
}

ExtractSIFT::ExtractSIFT() : detector(cv::xfeatures2d::SiftFeatureDetector::create(500)) {}

ordered_mat ExtractSIFT::operator()(const ordered_umat& image) const
{
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    detector->detectAndCompute(image.data, cv::noArray(), keyPoints, descriptors);

    return {image.rank, descriptors};
}


cv::Mat ExtractSIFT::operator()(const cv::UMat& image) const
{
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);

    return descriptors;
}

Frame ExtractSIFT::withKeyPoints(const cv::UMat& image) const
{
    Frame frame{};
    detector->detectAndCompute(image, cv::noArray(), frame.keyPoints, frame.descriptors);

    return frame;
}

ordered_mat ExtractColorHistogram::operator()(const ordered_umat& image) const
{
    return {image.rank, operator()(image.data)};
}

cv::Mat ExtractColorHistogram::operator()(const cv::UMat& image) const
{
    cv::Mat colorHistogram;
    cv::UMat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<int> histSize{HBINS, SBINS};
    // hue varies from 0 to 179, see cvtColor
    std::vector<float> ranges{0, 180, 0, 256};
    // we compute the histogram from the 0-th and 1-st channels
    std::vector<int> channels{0, 1};

    cv::calcHist(std::vector<decltype(hsv)>{hsv}, channels, cv::Mat(), // do not use mask
                 colorHistogram, histSize, ranges,
                 true);
    cv::normalize(colorHistogram, colorHistogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return colorHistogram;
}

ExtractFrame::ExtractFrame(const Vocab<Frame> &frameVocab) : vocab(frameVocab) {}

ordered_mat ExtractFrame::operator()(const ordered_mat& frame) const
{
    return {frame.rank, operator()(frame.data)};
}

cv::Mat ExtractFrame::operator()(const cv::Mat& frame) const
{
    return baggify(frame, BOWExtractor{vocab});
}

void SaveFrameSink::operator()(const ordered_frame& frame) const
{
    if(frame.rank % 40 == 0)
        std::cout << "frame: " << frame.rank << std::endl;

    loader.saveFrame(video, frame.rank, frame.data);
}
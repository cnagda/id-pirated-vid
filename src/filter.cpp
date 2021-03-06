#include "filter.hpp"
#include "imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "matcher.hpp"
#include <cassert>
#include <iostream>
#include "vocabulary.hpp"

ordered_umat ScaleImage::operator()(const ordered_umat& image) const
{
    return {image.rank, scaleToTarget(image.data, cropsize.first, cropsize.second)};
}

cv::UMat ScaleImage::operator()(const cv::UMat& image) const {
    return scaleToTarget(image, cropsize.first, cropsize.second);
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

ExtractAKAZE::ExtractAKAZE() : detector(cv::AKAZE::create()) {}

ordered_mat Extract2DColorHistogram::operator()(const ordered_umat& image) const
{
    return {image.rank, operator()(image.data)};
}

cv::Mat Extract2DColorHistogram::operator()(const cv::UMat& image) const
{
    cv::Mat colorHistogram;
    cv::UMat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_RGB2HSV);

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

ordered_mat ExtractLUVColorHistogram::operator()(const ordered_umat& image) const
{
    return {image.rank, operator()(image.data)};
}

cv::Mat ExtractLUVColorHistogram::operator()(const cv::UMat& image) const
{
    cv::Mat colorHistogram;
    cv::UMat lab;
    cv::cvtColor(image, lab, cv::COLOR_RGB2Luv);

    std::vector<int> histSize{8, 8, 8};
    // hue varies from 0 to 179, see cvtColor
    std::vector<float> ranges{0, 100, -127, 127, -127, 127};
    // we compute the histogram from the 0-th and 1-st channels
    std::vector<int> channels{0, 1, 2};

    cv::calcHist(std::vector{lab}, channels, cv::Mat(), // do not use mask
                 colorHistogram, histSize, ranges,
                 true);

    cv::normalize(colorHistogram, colorHistogram, 1, 0, cv::NORM_L1);

    return colorHistogram;
}

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
    loader.saveFrame(video, frame.rank, frame.data);
}

#include "filter.hpp"
#include "imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "vocabulary.hpp"
#include "matcher.hpp"
#include <cassert>

#define HBINS 32
#define SBINS 30

ordered_mat VideoFrameSource::operator()(tbb::flow_control& fc)
{
    ordered_mat image;

    if (cap.read(image.data))
    {
        image.rank = counter++;
        return image;
    }

    fc.stop();
    return image;
}

ordered_mat& ScaleImage::operator()(ordered_mat& image) const
{
    image.data = scaleToTarget(image.data, cropsize.first, cropsize.second);
    return image;
}

ExtractSIFT::ExtractSIFT() : detector(cv::xfeatures2d::SiftFeatureDetector::create(500)) {}

ordered_mat ExtractSIFT::operator()(const ordered_mat& image) const
{
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keyPoints;
    detector->detectAndCompute(image.data, cv::noArray(), keyPoints, descriptors);

    return {image.rank, descriptors};
}

ordered_mat ExtractColorHistogram::operator()(const ordered_mat& image) const
{
    cv::Mat colorHistogram;
    cv::UMat hsv;
    cv::cvtColor(image.data, hsv, cv::COLOR_BGR2HSV);

    std::vector<int> histSize{HBINS, SBINS};
    // hue varies from 0 to 179, see cvtColor
    std::vector<float> ranges{0, 180, 0, 256};
    // we compute the histogram from the 0-th and 1-st channels
    std::vector<int> channels{0, 1};

    cv::calcHist(std::vector<decltype(hsv)>{hsv}, channels, cv::Mat(), // do not use mask
                 colorHistogram, histSize, ranges,
                 true);
    cv::normalize(colorHistogram, colorHistogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return {image.rank, colorHistogram};
}

ordered_mat ExtractFrame::operator()(const ordered_mat& frame) const
{
    return {frame.rank, baggify(frame.data, frameVocab.descriptors())};
}

void SaveFrameSink::operator()(const ordered_frame& frame) const
{
    if(frame.rank % 40 == 0)
        std::cout << "frame: " << frame.rank << std::endl;

    loader.saveFrame(video, frame.rank, frame.data);
}
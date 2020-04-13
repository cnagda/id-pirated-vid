#ifndef FRAME_H
#define FRAME_H
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <algorithm>

inline bool keyPointEqual(const cv::KeyPoint& a, const cv::KeyPoint& b) {
    return a.size == b.size &&
        a.angle == b.angle &&
        a.class_id == b.class_id &&
        a.class_id == b.class_id &&
        a.pt.x == b.pt.x &&
        a.pt.y == b.pt.y &&
        a.response == b.response &&
        a.size == b.size;
}

inline bool matEqual(const cv::Mat& a, const cv::Mat& b) {
    return (a.size() == b.size()) && std::equal(a.begin<float>(), a.end<float>(), b.begin<float>());
}

class Frame {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors, frameDescriptor, colorHistogram;
    Frame() = default;
    Frame(const cv::Mat& descriptors,
        const cv::Mat& frameDescriptor, const cv::Mat& colorHistogram) :
        keyPoints(), descriptors(descriptors), frameDescriptor(frameDescriptor),
        colorHistogram(colorHistogram) {};

    template<typename Range> Frame(const Range& range, const cv::Mat& descriptors,
        const cv::Mat& frameDescriptor, const cv::Mat& colorHistogram) :
        Frame(descriptors, frameDescriptor, colorHistogram) {
        std::copy(std::begin(range), std::end(range), std::back_inserter(keyPoints));
    }

    bool operator==(const Frame& f2) const {
        return std::equal(keyPoints.begin(), keyPoints.end(), f2.keyPoints.begin(), keyPointEqual) &&
            matEqual(descriptors, f2.descriptors) &&
            matEqual(frameDescriptor, f2.frameDescriptor) &&
            matEqual(colorHistogram, f2.colorHistogram);
    }
    static const std::string vocab_name;
};

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);

#endif

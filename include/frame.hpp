#ifndef FRAME_H
#define FRAME_H
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

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
    return std::equal(a.begin<int>(), a.end<int>(), b.begin<int>());
}

class Frame {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors, frameDescriptor, colorHistogram;
    bool operator==(const Frame& f2) const {
        return std::equal(keyPoints.begin(), keyPoints.end(), f2.keyPoints.begin(), keyPointEqual) &&
            descriptors.size == f2.descriptors.size &&
            matEqual(descriptors, f2.descriptors) &&
            matEqual(frameDescriptor, f2.frameDescriptor) &&
            matEqual(colorHistogram, f2.colorHistogram);
    }
    static const std::string vocab_name;
};

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);

#endif

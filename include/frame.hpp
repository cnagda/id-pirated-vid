#ifndef FRAME_H
#define FRAME_H
#include <vector>
#include <opencv2/opencv.hpp>

inline bool compareKeyPoint(cv::KeyPoint a, cv::KeyPoint b) {
    return a.size == b.size &&
        a.angle == b.angle &&
        a.class_id == b.class_id &&
        a.class_id == b.class_id &&
        a.pt.x == b.pt.x &&
        a.pt.y == b.pt.y &&
        a.response == b.response &&
        a.size == b.size;
}

class Frame {
public:
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    bool operator==(const Frame& f2) const {
        return std::equal(keyPoints.begin(), keyPoints.end(), f2.keyPoints.begin(), compareKeyPoint) &&
            descriptors.size == f2.descriptors.size &&
            std::equal(descriptors.begin<float>(), descriptors.end<float>(), f2.descriptors.begin<float>());
    }
    static const std::string vocab_name;
};

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);

#endif
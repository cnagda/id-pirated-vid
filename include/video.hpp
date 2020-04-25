#ifndef VIDEO_HPP
#define VIDEO_HPP
#include <vector>
#include <string>
#include <functional>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <algorithm>
#include <memory>
#include "database_iface.hpp"

#define HBINS 32
#define SBINS 30

inline bool keyPointEqual(const cv::KeyPoint &a, const cv::KeyPoint &b)
{
    return a.size == b.size &&
           a.angle == b.angle &&
           a.class_id == b.class_id &&
           a.class_id == b.class_id &&
           a.pt.x == b.pt.x &&
           a.pt.y == b.pt.y &&
           a.response == b.response &&
           a.size == b.size;
}

inline bool matEqual(const cv::Mat &a, const cv::Mat &b)
{
    return (a.size() == b.size()) && (a.empty() || std::equal(a.begin<float>(), a.end<float>(), b.begin<float>()));
}

struct Frame;
struct SerializableScene;

struct SIFTVideo
{
    const std::string filename;
    const std::function<void(cv::UMat, Frame)> callback;
    const std::pair<int, int> cropsize;

    SIFTVideo(const std::string &filename, std::function<void(cv::UMat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700})
        : filename(filename), callback(callback), cropsize(cropsize){};
        
    SIFTVideo(SIFTVideo &&) = default;

    std::unique_ptr<ICursor<Frame>> frames() const;
    std::unique_ptr<ICursor<cv::UMat>> images() const;
};

struct Frame
{
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors, frameDescriptor, colorHistogram;
    Frame() = default;
    Frame(const cv::Mat &descriptors,
          const cv::Mat &frameDescriptor, const cv::Mat &colorHistogram) : keyPoints(), descriptors(descriptors), frameDescriptor(frameDescriptor),
                                                                           colorHistogram(colorHistogram){};

    template <typename Range>
    Frame(const Range &range, const cv::Mat &descriptors,
          const cv::Mat &frameDescriptor, const cv::Mat &colorHistogram) : Frame(descriptors, frameDescriptor, colorHistogram)
    {
        std::copy(std::begin(range), std::end(range), std::back_inserter(keyPoints));
    }

    bool operator==(const Frame &f2) const
    {
        return std::equal(keyPoints.begin(), keyPoints.end(), f2.keyPoints.begin(), f2.keyPoints.end(), keyPointEqual) &&
               matEqual(descriptors, f2.descriptors) &&
               matEqual(frameDescriptor, f2.frameDescriptor) &&
               matEqual(colorHistogram, f2.colorHistogram);
    }
    static const std::string vocab_name;
};

struct SerializableScene
{
    cv::Mat frameBag;
    size_t startIdx, endIdx;
    const static std::string vocab_name;

    SerializableScene() : frameBag(), startIdx(), endIdx(){};
    SerializableScene(size_t startIdx, size_t endIdx) : startIdx(startIdx), endIdx(endIdx), frameBag(){};
    SerializableScene(const cv::Mat &matrix, size_t startIdx, size_t endIdx) : startIdx(startIdx), endIdx(endIdx), frameBag(matrix){};
};

SIFTVideo getSIFTVideo(const std::string &filename, std::function<void(cv::UMat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});

bool SceneWrite(const std::string &filename, const SerializableScene &frame);
SerializableScene SceneRead(const std::string &filename);

bool SIFTwrite(const std::string &filename, const Frame &frame);
Frame SIFTread(const std::string &filename);

#endif
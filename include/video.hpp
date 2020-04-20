#ifndef VIDEO_HPP
#define VIDEO_HPP
#include <vector>
#include <string>
#include <functional>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <algorithm>

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
    using size_type = std::vector<Frame>::size_type;

    std::vector<Frame> SIFTFrames;
    SIFTVideo() : SIFTFrames(){};
    SIFTVideo(const std::vector<Frame> &frames) : SIFTFrames(frames){};
    SIFTVideo(std::vector<Frame> &&frames) : SIFTFrames(frames){};
    std::vector<Frame> &frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

typedef SIFTVideo::size_type v_size;

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
    v_size startIdx, endIdx;
    const static std::string vocab_name;

    SerializableScene() : frameBag(), startIdx(), endIdx(){};
    SerializableScene(v_size startIdx, v_size endIdx) : startIdx(startIdx), endIdx(endIdx), frameBag(){};
    SerializableScene(const cv::Mat &matrix, v_size startIdx, v_size endIdx) : startIdx(startIdx), endIdx(endIdx), frameBag(matrix){};

    template <typename Video>
    inline auto getFrameRange(Video &&video) const
    {
        auto &frames = video.frames();
        return getFrameRange(frames.begin(),
                             typename std::iterator_traits<decltype(frames.begin())>::iterator_category());
    }

    template <typename It>
    inline auto getFrameRange(It begin, std::random_access_iterator_tag) const
    {
        return std::make_pair(begin + startIdx, begin + endIdx);
    }
};

SIFTVideo getSIFTVideo(const std::string &filename, std::function<void(cv::UMat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});

bool SceneWrite(const std::string &filename, const SerializableScene &frame);
SerializableScene SceneRead(const std::string &filename);

bool SIFTwrite(const std::string &filename, const Frame &frame);
Frame SIFTread(const std::string &filename);

#endif
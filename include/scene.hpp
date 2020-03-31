#ifndef SCENE_HPP
#define SCENE_HPP
#include "frame.hpp"
#include "video.hpp"
#include <string>

struct SerializableScene {
    cv::Mat frameBag;
    SIFTVideo::size_type startIdx, endIdx;
    SIFTVideo::size_type sceneNumber;
    const static std::string vocab_name;

    explicit SerializableScene(): frameBag(), startIdx(), endIdx() {};
    explicit SerializableScene(SIFTVideo::size_type startIdx, SIFTVideo::size_type endIdx) :
        startIdx(startIdx), endIdx(endIdx), frameBag() {};
    explicit SerializableScene(const cv::Mat& matrix, SIFTVideo::size_type startIdx, SIFTVideo::size_type endIdx) :
        startIdx(startIdx), endIdx(endIdx), frameBag(matrix) {};

    template<typename Video>
    inline auto getFrameRange(Video&& video) const {
        auto& frames = video.frames();
        return getFrameRange(frames.begin(), 
        typename std::iterator_traits<decltype(frames.begin())>::iterator_category());
    }

    template<typename It>
    inline auto getFrameRange(It begin, std::random_access_iterator_tag) const {
        return std::make_pair(begin + startIdx, begin + endIdx);
    }
};

void SceneWrite(const std::string& filename, const SerializableScene& frame);
SerializableScene SceneRead(const std::string& filename);

#endif
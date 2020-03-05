#ifndef SCENE_HPP
#define SCENE_HPP
#include "frame.hpp"
#include "video.hpp"
#include "vocabulary.hpp"
#include <string>

struct SerializableScene {
    cv::Mat frameBag;
    SIFTVideo::size_type startIdx, endIdx;
    const static std::string vocab_name;

    explicit SerializableScene(SIFTVideo::size_type startIdx, SIFTVideo::size_type endIdx) :
        startIdx(startIdx), endIdx(endIdx), frameBag() {};
    explicit SerializableScene(const cv::Mat& matrix, SIFTVideo::size_type startIdx, SIFTVideo::size_type endIdx) :
        startIdx(startIdx), endIdx(endIdx), frameBag(matrix) {};

    template<typename Video>
    auto getFrameRange(Video& video) const {
        auto& frames = video.frames();
        return getFrameRange(frames.begin(), 
        typename std::iterator_traits<decltype(frames.begin())>::iterator_category());
    };

    template<typename It>
    auto getFrameRange(It begin, std::random_access_iterator_tag) const {
        return std::make_pair(begin + startIdx, begin + endIdx);
    };

    template<typename Video, typename DB>
    const cv::Mat& descriptor(Video&& video, DB&& database) & {
        if(frameBag.empty()) {
            auto& frames = video.frames();
            auto vocab = loadVocabulary<Vocab<Frame>>(database);
            auto frameVocab = loadVocabulary<Vocab<SerializableScene>>(database);
            if(!vocab | !frameVocab) {
                throw std::runtime_error("Scene couldn't get a frame vocabulary");
            }
            auto access = [vocab = vocab->descriptors()](auto frame){ return baggify(frame.descriptors, vocab); };
            frameBag = baggify(
                boost::make_transform_iterator(frames.begin(), access),
                boost::make_transform_iterator(frames.end(), access),
                frameVocab->descriptors());
        }

        return frameBag;
    }
};

void SceneWrite(const std::string& filename, const SerializableScene& frame);
SerializableScene SceneRead(const std::string& filename);

#endif
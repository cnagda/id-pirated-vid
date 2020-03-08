#ifndef KEYFRAMES_HPP
#define KEYFRAMES_HPP

#include "concepts.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <boost/iterator/transform_iterator.hpp>
#include <type_traits>
#include "frame.hpp"
#include "vocab_type.hpp"
#include "scene.hpp"

class SerializableScene;
class FileDatabase;

#define FRAMES_PER_SCENE  45

template <class T, class RankType>
struct sortable{
    RankType rank;
    T data;
    bool operator<(const sortable& a) const {  return rank < a.rank; };
};


template<typename Matrix>
cv::Mat constructVocabulary(Matrix&& descriptors, unsigned int K, cv::Mat labels = cv::Mat()) {
	//cv::BOWKMeansTrainer trainer(K);
    cv::Mat retval;

    kmeans(descriptors, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, retval);

    std::cout << "About to return" << std::endl;

    return retval;
	//return trainer.cluster(descriptors);
}

template<typename It>
cv::Mat constructVocabulary(It start, It end, unsigned int K, cv::Mat labels = cv::Mat()) {
	cv::Mat accumulator;
    for(auto i = start; i != end; ++i)
        accumulator.push_back(*i);
    return constructVocabulary(accumulator, K, labels);
}

Vocab<Frame> constructFrameVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator = 1);
Vocab<SerializableScene> constructSceneVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator = 1);

template<typename Matrix, typename Vocab>
cv::Mat baggify(Matrix&& f, Vocab&& vocab) {
    cv::BOWImgDescriptorExtractor extractor(cv::FlannBasedMatcher::create());

    if constexpr(std::is_invocable_v<Vocab>) {
        extractor.setVocabulary(vocab());
    } else {
        extractor.setVocabulary(vocab);
    }

    cv::Mat output;

    if(!f.empty()){
        extractor.compute(f, output);
    }
    else{
        // std::cerr << "In baggify: Frame dimension does not match vocab" << std::endl;
    }

    return output;
}

template<typename It, typename Vocab>
cv::Mat baggify(It rangeBegin, It rangeEnd, Vocab&& vocab) {
    cv::Mat accumulator;
    for(auto i = rangeBegin; i != rangeEnd; ++i)
        accumulator.push_back(*i);
    return baggify(accumulator, std::forward<Vocab>(vocab));
}

template<typename It, typename Vocab>
inline cv::Mat baggify(std::pair<It, It> pair, Vocab&& vocab) {
    return baggify(pair.first, pair.second, std::forward<Vocab>(vocab));
}

template<class Video, typename Cmp>
auto flatScenes(Video& video, Cmp&& comp, double threshold){
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto& frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    index_t last = 0;

    // for(int i = 1; i < frames.size(); i++) {
    //     if(double fs = comp(frames[i], frames[i - 1]); fs < threshold){
    //         if(!frames[i].descriptors.empty()){ // do not include black frames
    //             retval.push_back({last, i});
    //             last = i;
    //         }
    //     } else {
    //         // std::cout << "i: " << i << "sim: " << fs << std::endl;
    //     }
    // }

    for(int i = FRAMES_PER_SCENE - 1; i < frames.size() - FRAMES_PER_SCENE; i+=FRAMES_PER_SCENE) {
        while (frames[i].descriptors.empty() && i < frames.size() - FRAMES_PER_SCENE) { i++; }
        if (i == frames.size()) { break; }
        retval.push_back({last, i});
        last = i;
    }
    if(!frames.back().descriptors.empty()){ // do not include black frames
        retval.push_back({last, frames.size()});
    }

    return retval;
}

template<typename RangeIt, typename Vocab>
std::enable_if_t<is_pair_iterator_v<RangeIt> &&
    !std::is_integral_v<decltype(std::declval<RangeIt>()->first)>, std::vector<cv::Mat>>
flatScenesBags(RangeIt start, RangeIt end, Vocab&& frameVocab) {
    std::cout << "In flatScenesBags" << std::endl;

    std::vector<cv::Mat> retval2;

    for(auto i = start; i < end; i++){
        retval2.push_back(baggify(*i, frameVocab));
    }
    return retval2;
}

template<class Video, typename IndexIt, typename Vocab>
std::enable_if_t<is_pair_iterator_v<IndexIt> &&
    std::is_integral_v<decltype(std::declval<IndexIt>()->first)>, std::vector<cv::Mat>>
flatScenesBags(Video& video, IndexIt start, IndexIt end, Vocab&& vocab, Vocab&& frameVocab){
    static_assert(is_pair_iterator_v<IndexIt>,
        "flatScenesBags requires an iterator to a pair");

    auto accessor = [vocab](const Frame& frame) { return baggify(frame.descriptors, vocab); };
    auto& frames = video.frames();
    auto begin = frames.begin();

    auto func = [begin, accessor](auto i) {
        return std::make_pair(
            boost::make_transform_iterator(begin + i.first, accessor),
            boost::make_transform_iterator(begin + i.second, accessor));
    };

    return flatScenesBags(boost::make_transform_iterator(start, func),
        boost::make_transform_iterator(end, func), frameVocab);
}

template<class Video, typename Cmp, typename Vocab>
inline std::vector<cv::Mat> flatScenesBags(Video &video, Cmp&& comp, double threshold, Vocab&& vocab, Vocab&& frameVocab) {
    auto ss = flatScenes(video, comp, threshold);
    return flatScenesBags(video, ss.begin(), ss.end(), frameVocab);
}

template<typename V, typename Db>
bool saveVocabulary(V&& vocab, Db&& db) {
    return db.saveVocab(std::forward<V>(vocab), std::remove_reference_t<V>::vocab_name);
}

template<typename V, typename Db>
std::optional<V> loadVocabulary(Db&& db) {
    auto v = db.loadVocab(V::vocab_name);
    if(v) {
        return V(v.value());
    }
    return std::nullopt;
}

template<typename V, typename Db>
std::optional<V> loadOrComputeVocab(Db&& db, int K) {
    auto vocab = loadVocabulary<V>(std::forward<Db>(db));
    if(!vocab) {
        if(K == -1) {
            return std::nullopt;
        }

        V v;
        if constexpr(std::is_same_v<typename V::vocab_type, Frame>) {
            v = constructFrameVocabulary(db, K, 10);
        } else if constexpr(std::is_base_of_v<typename V::vocab_type, SerializableScene>) {
            v = constructSceneVocabulary(db, K);
        }
        saveVocabulary(std::forward<V>(v), std::forward<Db>(db));
        return v;
    }
    return vocab.value();
}


template<class Vocab>
cv::Mat loadFrameDescriptor(Frame& frame, Vocab&& vocab) {
    if(frame.frameDescriptor.empty()) {
        frame.frameDescriptor = getFrameDescriptor(frame, std::forward<Vocab>(vocab));
    }
    return frame.frameDescriptor;
}

template<class Vocab>
inline cv::Mat getFrameDescriptor(const Frame& frame, Vocab&& vocab) {   
    return baggify(frame.descriptors, std::forward<Vocab>(vocab));
}

template<class Video, class DB>
cv::Mat getSceneDescriptor(const SerializableScene& scene, Video&& video, DB&& database) {
    auto frames = scene.getFrameRange(std::forward<Video>(video));
    auto vocab = loadVocabulary<Vocab<Frame>>(std::forward<DB>(database));
    auto frameVocab = loadVocabulary<Vocab<SerializableScene>>(std::forward<DB>(database));
    if(!vocab | !frameVocab) {
        return scene.frameBag;
    }
    auto access = [vocab = vocab->descriptors()](auto frame){ return loadFrameDescriptor(frame, vocab); };
    return baggify(
        boost::make_transform_iterator(frames.first, access),
        boost::make_transform_iterator(frames.second, access),
        frameVocab->descriptors());
}

template<class Video, class DB>
cv::Mat loadSceneDescriptor(SerializableScene& scene, Video&& video, DB&& db) {
    if(scene.frameBag.empty()) {
        scene.frameBag = getSceneDescriptor(scene, std::forward<Video>(video), std::forward<DB>(db));
    }
    
    return scene.frameBag;
}


template<typename Vocab> class BOWComparator {
    static_assert(std::is_constructible_v<Vocab, Vocab>,
                  "Vocab must be constructible");
    const Vocab vocab;
public:
    BOWComparator(const Vocab& vocab) : vocab(vocab) {};
    double operator()(Frame& f1, Frame& f2) const {
        return frameSimilarity(f1, f2, [this](Frame& f){ return loadFrameDescriptor(f, vocab); });
    }
};

#endif

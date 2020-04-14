#ifndef KEYFRAMES_HPP
#define KEYFRAMES_HPP

#include "concepts.hpp"
#include <iostream>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <boost/iterator/transform_iterator.hpp>
#include <type_traits>
#include "frame.hpp"
#include "vocab_type.hpp"
#include "scene.hpp"

struct SerializableScene;
class FileDatabase;

#define FRAMES_PER_SCENE  45


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

    cv::UMat copy;
    accumulator.copyTo(copy);

    return constructVocabulary(copy, K, labels);
}

Vocab<Frame> constructFrameVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator = 1);
Vocab<SerializableScene> constructSceneVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator = 1);
Vocab<Frame> constructFrameVocabularyHierarchical(const FileDatabase& database, unsigned int K, unsigned int N, unsigned int speedinator);

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

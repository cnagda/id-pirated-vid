#ifndef KEYFRAMES_HPP
#define KEYFRAMES_HPP

#include "concepts.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <boost/iterator/transform_iterator.hpp>
#include <type_traits>
#include "frame.hpp"

class SerializableScene;
class FileDatabase;

template <class T, class RankType>
struct sortable{
    RankType rank;
    T data;
    bool operator<(const sortable& a) const {  return rank < a.rank; };
};


class ContainerVocab {
private:
    cv::Mat desc;
    std::string hash;
public:
    ContainerVocab() = default;
    ContainerVocab(const cv::Mat& descriptors) : desc(descriptors), hash() {};
    std::string getHash() const {
        return hash;
    }
    cv::Mat descriptors() const {
        return desc;
    }
    static const std::string vocab_name;
};

template<typename T>
class Vocab : public ContainerVocab {
public:
    using ContainerVocab::ContainerVocab;
    Vocab(const ContainerVocab& v) : ContainerVocab(v) {};
    static const std::string vocab_name;
    typedef T vocab_type;
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
    return baggify(accumulator, vocab);
}

template<typename It, typename Vocab>
inline cv::Mat baggify(std::pair<It, It> pair, Vocab&& vocab) {
    return baggify(pair.first, pair.second, vocab);
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

    for(int i = 1; i < frames.size(); i++) {
        if(double fs = comp(frames[i], frames[i - 1]); fs < threshold){
            if(!frames[i].descriptors.empty()){ // do not include black frames
                retval.push_back({last, i});
                last = i;
            }
        } else {
            // std::cout << "i: " << i << "sim: " << fs << std::endl;
        }
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

void visualizeSubset(std::string fname, const std::vector<int>& subset = {});

template<typename RangeIt>
std::enable_if_t<is_pair_iterator_v<RangeIt>, void>
visualizeSubset(std::string fname, RangeIt begin, RangeIt end) {
    std::vector<int> subset;
    for(auto i = begin; i < end; i++)
        for(auto j = begin->first; j < begin->second; j++)
        subset.push_back(j);

    visualizeSubset(fname, subset);
}

template<typename It>
std::enable_if_t<!is_pair_iterator_v<It>, void>
visualizeSubset(std::string fname, It begin, It end) {
    auto size = std::distance(begin, end);
    std::cout << "In visualise subset" << std::endl;

    using namespace cv;

    namedWindow("Display window", WINDOW_NORMAL );

    VideoCapture cap(fname, CAP_ANY);

    int count = -1;
    int index = 0;
    Mat image;

    while(cap.read(image)){
        ++count;
        if(size && count != begin[index]){
            continue;
        }

        index++;
        if(index >= size){
            index = 0;
        }
        imshow("Display window", image);
        waitKey(0);
    };
    destroyWindow("Display window");
}


inline void visualizeSubset(std::string fname, const std::vector<int>& subset){
    visualizeSubset(fname, subset.begin(), subset.end());
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

#endif

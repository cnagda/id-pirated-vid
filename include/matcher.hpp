#ifndef BOW_HPP
#define BOW_HPP

#include "instrumentation.hpp"
#include "database.hpp"
#include "matrix.hpp"
#include <opencv2/opencv.hpp>
#include <optional>
#include <iterator>
#include <type_traits>
#include <exception>

struct MatchInfo {
    double matchConfidence;
    IVideo::size_type startFrame, endFrame;
    std::string video;
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

Vocab<Frame> constructFrameVocabulary(const IDatabase& database, unsigned int K, unsigned int speedinator = 1);

Vocab<IScene> constructSceneVocabulary(const IDatabase& database, unsigned int K, unsigned int speedinator = 1);

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

template<typename Matrix>
double cosineSimilarity(Matrix&& b1, Matrix&& b2) {
    if(b1.empty() || b1.size() != b2.size()) return -1;

    auto b1n = b1.dot(b1);
    auto b2n = b2.dot(b2);

    return b1.dot(b2)/(sqrt(b1n * b2n) + 1e-10);
}

template<typename Extractor> 
double frameSimilarity(const Frame& f1, const Frame& f2, Extractor&& extractor){
    auto b1 = extractor(f1), b2 = extractor(f2);

    return cosineSimilarity(b1, b2);
}

template<typename Vocab> class BOWComparator {
    static_assert(std::is_constructible_v<Vocab, Vocab>,
                  "Vocab must be constructible");
    const Vocab vocab;
public:
    BOWComparator(const Vocab& vocab) : vocab(vocab) {};
    double operator()(const Frame& f1, const Frame& f2) const {
        return frameSimilarity(f1, f2, [this](const Frame& f){ return baggify(f.descriptors, vocab); });
    }
};

double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter = nullptr);
std::optional<MatchInfo> findMatch(IVideo& target, IDatabase& db);

#endif

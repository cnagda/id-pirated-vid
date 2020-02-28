#ifndef BOW_HPP
#define BOW_HPP

#include "instrumentation.hpp"
#include "database_iface.hpp"
#include "vocabulary.hpp"
#include "matrix.hpp"
#include "sw.hpp"
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

template<class Video>
double boneheadedSimilarity(Video& v1, Video& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter){
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    int len = std::min(frames1.size(), frames2.size());

    for(int i = 0; i < len; i++){
        auto t = comparator(frames1[i], frames2[i]);
        if(reporter) reporter(FrameSimilarityInfo{t, frames1[i], frames2[i], i, i, 
            std::make_optional(std::ref(v1)), std::make_optional(std::ref(v2))});

        total += (t != -1)? t : 0;
    }

    return total/len;
}

template<class Video>
std::optional<MatchInfo> findMatch(Video& target, FileDatabase& db) {
    auto vocab = loadVocabulary<Vocab<Frame>>(db)->descriptors();
    auto frameVocab = loadVocabulary<Vocab<IScene>>(db)->descriptors();
    auto videopaths = db.loadVideo();

    auto frameComp = BOWComparator(vocab);
    
    auto intcomp = [](auto f1, auto f2) { return cosineSimilarity(f1, f2) > 0.8 ? 3 : -3; };

    MatchInfo match;
    auto targetFrames = target.frames();
    auto targetScenes = flatScenesBags(target, frameComp, 0.2f, frameVocab);

    for(auto& v2 : videopaths) {
        std::cout << "Calculating match for " << v2->name << std::endl;
        auto knownScenes = flatScenesBags(*v2, frameComp, 0.2f, frameVocab);

        auto&& alignments = calculateAlignment(targetScenes, knownScenes, intcomp, 0, 2);
        if(alignments.size() > 0) {
            auto& a = alignments[0];
            if(a.score > match.matchConfidence) {
                match = MatchInfo{a.score, a.startKnown, a.endKnown, v2->name};
            }
        }
        
    }
    
    if(match.matchConfidence > 0.5) {
        return match;
    }

    return std::nullopt;
}

#endif

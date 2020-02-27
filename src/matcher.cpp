#include <fstream>
#include <iostream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"
#include "kmeans2.hpp"
#include "sw.hpp"
#include "keyframes.hpp"

Vocab<Frame> constructFrameVocabulary(const IDatabase& database, unsigned int K, unsigned int speedinator) {
    cv::Mat descriptors;

    for(auto &video : database.loadVideo()) {
        auto frames = video->frames();
        for(auto i = frames.begin(); i != frames.end(); i += speedinator)
                descriptors.push_back(i->descriptors);
    }

    return constructVocabulary(descriptors, K);
}

Vocab<IScene> constructSceneVocabulary(const IDatabase& database, unsigned int K, unsigned int speedinator) {
    auto vocab = database.loadVocab<Vocab<Frame>>().descriptors();
    if(vocab.empty()) {
        throw std::runtime_error("trying to construct frame vocab but sift vocab is empty");
    }

    cv::Mat descriptors;

    for(auto &video : database.loadVideo()) {
        auto frames = video->frames();
        for(auto i = frames.begin(); i != frames.end(); i += speedinator)
                descriptors.push_back(baggify(i->descriptors, vocab));
    }

    return constructVocabulary(descriptors, K);
}

double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter){
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    int len = std::min(frames1.size(), frames2.size());

    for(int i = 0; i < len; i++){
        auto t = comparator(frames1[i], frames2[i]);
        if(reporter) reporter(FrameSimilarityInfo{t, frames1[i], frames2[i], i, i, &v1, &v2});

        total += (t != -1)? t : 0;
    }

    return total/len;
}

std::optional<MatchInfo> findMatch(IVideo& target, IDatabase& db, const cv::Mat& vocab, const cv::Mat& frameVocab) {
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

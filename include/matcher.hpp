#ifndef BOW_HPP
#define BOW_HPP

#include "instrumentation.hpp"
#include "database_iface.hpp"
#include <opencv2/opencv.hpp>
#include "sw.hpp"
#include <optional>

struct MatchInfo {
    double matchConfidence;
    IVideo::size_type startFrame, endFrame;
    std::string video;
    std::vector<Alignment> alignments;
};

template<typename Matrix>
double cosineSimilarity(Matrix&& b1, Matrix&& b2) {
    if(b1.empty() || b1.size() != b2.size()) return -1;

    auto b1n = b1.dot(b1);
    auto b2n = b2.dot(b2);

    return b1.dot(b2)/(sqrt(b1n * b2n) + 1e-10);
}

template<typename Extractor>
double frameSimilarity(Frame& f1, Frame& f2, Extractor&& extractor){
    auto b1 = extractor(f1), b2 = extractor(f2);

    return cosineSimilarity(b1, b2);
}

class ColorComparator {
public:
    double operator()(const Frame& f1, const Frame& f2) const;
};

double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter);

std::optional<MatchInfo> findMatch(IVideo& target, FileDatabase& db);

#endif

#ifndef BOW_HPP
#define BOW_HPP

#include "instrumentation.hpp"
#include "database.hpp"
#include <opencv2/opencv.hpp>
#include <optional>

struct MatchInfo {
    double matchConfidence;
    IVideo::size_type startFrame, endFrame;
    std::string video;
};

cv::Mat constructVocabulary(const std::string& path, int K = -1, int speedinator = 1);
cv::Mat constructMyVocabulary(const std::string& path, int K = -1, int speedinator = 1);
cv::Mat baggify(Frame f, cv::Mat vocab);

double frameSimilarity(Frame f1, Frame f2, std::function<cv::Mat(Frame)> extractor);
double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter = nullptr);
std::optional<MatchInfo> findMatch(IVideo& target, IDatabase& db, cv::Mat vocab);

#endif

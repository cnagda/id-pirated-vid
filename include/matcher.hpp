#ifndef BOW_HPP
#define BOW_HPP

#include "instrumentation.hpp"
#include "database_iface.hpp"
#include <vector>

struct MatchInfo
{
    std::string video;
    double confidence;
    IVideo::size_type startQuery, endQuery, startKnown, endKnown;
};

template <typename Matrix>
double cosineSimilarity(Matrix &&b1, Matrix &&b2)
{
    if (b1.empty() || b1.size() != b2.size())
        return -1;

    auto b1n = b1.dot(b1);
    auto b2n = b2.dot(b2);

    return b1.dot(b2) / (sqrt(b1n * b2n) + 1e-10);
}

template <typename Extractor>
double frameSimilarity(Frame &f1, Frame &f2, Extractor &&extractor)
{
    auto b1 = extractor(f1), b2 = extractor(f2);

    return cosineSimilarity(b1, b2);
}

namespace cv
{
class Mat;
}

class ColorComparator2D
{
public:
    double operator()(const Frame &f1, const Frame &f2) const;
    double operator()(const cv::Mat &, const cv::Mat &) const;
};


class ColorIntersectComparator
{
public:
    double operator()(const Frame &f1, const Frame &f2) const;
    double operator()(const cv::Mat &, const cv::Mat &) const;
};

class QueryVideo : public IVideo
{
    std::unique_ptr<ICursor<SerializableScene>> scenes;

public:
    QueryVideo(const IVideo &source, std::unique_ptr<ICursor<SerializableScene>> scenes)
        : IVideo(source), scenes(std::move(scenes)){};

    std::unique_ptr<ICursor<SerializableScene>> getScenes() { return std::move(scenes); };
};

class FileDatabase;
class DatabaseVideo;
class SIFTVideo;

QueryVideo make_query_adapter(const SIFTVideo&, const FileDatabase&);
QueryVideo make_query_adapter(const DatabaseVideo&);

std::vector<MatchInfo> findMatch(QueryVideo& target, const FileDatabase &db);
std::vector<MatchInfo> findMatch(QueryVideo&& target, const FileDatabase &db);
std::vector<MatchInfo> findMatch(std::unique_ptr<ICursor<Frame>>, const FileDatabase &db);
std::vector<MatchInfo> findMatch(std::unique_ptr<ICursor<SerializableScene>>, const FileDatabase &db);

#endif

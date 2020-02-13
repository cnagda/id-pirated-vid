#ifndef INSTRUMENTATION_HPP
#define INSTRUMENTATION_HPP
#include <string>
#include <vector>
#include <memory>
#include "database.hpp"
#include "frame.hpp"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

template <typename T> using optional_ref = std::optional<std::reference_wrapper<T>>;

struct FrameSimilarityInfo {
public:
    double similarity;
    Frame f1, f2;
    IVideo::size_type f1Idx, f2Idx;
    const IVideo* v1;
    const IVideo* v2;
};

typedef const std::string& Label;
typedef std::function<void(FrameSimilarityInfo)> SimilarityReporter;

struct Point2f {
    float x, y;
};

struct TimeSeries {
    Label name;
    std::vector<Point2f> data;
};

class IExporter {
public:
    virtual void exportTimeseries(Label title, Label xaxis, Label yaxis, const std::vector<TimeSeries>& data) {};
};

class VideoMatchingInstrumenter {
private:
    const IVideo& target;
    std::unordered_map<std::string, std::vector<Point2f>> videoTracker;
public:
    VideoMatchingInstrumenter(const IVideo &targetVideo) : target(targetVideo) {}
    void clear();
    void addFrameSimilarity(FrameSimilarityInfo info);
    std::vector<TimeSeries> getTimeSeries() const;
};

SimilarityReporter getReporter(VideoMatchingInstrumenter& instrumenter);

class CSVExporter : public IExporter {
public:
    void exportTimeseries(Label title, Label xaxis, Label yaxis, const std::vector<TimeSeries>& series) override;
};

class EmmaExporter : public IExporter {
private:
    const fs::path outputDir;
public:
    EmmaExporter(fs::path dir = fs::current_path() / "Temp") : outputDir(dir) {};
    void exportTimeseries(Label title, Label xaxis, Label yaxis, const std::vector<TimeSeries>& series) override; 
};

#endif
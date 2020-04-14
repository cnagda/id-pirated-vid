#ifndef INSTRUMENTATION_HPP
#define INSTRUMENTATION_HPP
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "database_iface.hpp"
#include <experimental/filesystem>
#include <functional>

namespace fs = std::experimental::filesystem;

template <typename T> using optional_ref = std::optional<std::reference_wrapper<T>>;

struct FrameSimilarityInfo {
    double similarity;
    IVideo::size_type f1Idx, f2Idx;
    const optional_ref<IVideo> v1;
    const optional_ref<IVideo> v2;
};

typedef std::string Label;
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
    virtual void exportTimeseries(const Label& title, const Label& xaxis, const Label& yaxis, const std::vector<TimeSeries>& data) const = 0;
};

template<class Video>
class VideoMatchingInstrumenter {
private:
    const Video& target;
    std::unordered_map<std::string, std::vector<Point2f>> videoTracker;
public:
    VideoMatchingInstrumenter(const Video &targetVideo) : target(targetVideo) {}
    void clear();
    void addFrameSimilarity(FrameSimilarityInfo info){
        if(info.v1 && info.v2) {
            auto& v1_ = info.v1->get();
            auto& v2_ = info.v2->get();
            auto& known = (v1_.name == target.name) ? v2_ : v1_ ;

            videoTracker[known.name].push_back({info.f1Idx, info.similarity}); 
        }
    }
    std::vector<TimeSeries> getTimeSeries() const {
        std::vector<TimeSeries> out;
        for(auto& [name, points] : videoTracker) {
            std::vector<Point2f> dataCopy(points);
            std::sort(dataCopy.begin(), dataCopy.end(), [](Point2f p1, Point2f p2){ return p1.x < p2.x; });
            out.push_back({name, dataCopy});
        }
    
        return out;
    }
};


template<class Video>
SimilarityReporter getReporter(VideoMatchingInstrumenter<Video>& instrumenter) {
    return [&instrumenter](auto f) { instrumenter.addFrameSimilarity(f); };
}

class FSExporter : public IExporter {
protected:
    const fs::path outputDir;
public:
    FSExporter(fs::path dir = fs::current_path() / "Temp") : outputDir(dir) { };
};

class CSVExporter : public FSExporter {
public:
    const std::string delimiter = ",";
    void exportTimeseries(const Label& title, const Label& xaxis, const Label& yaxis, const std::vector<TimeSeries>& data) const override;
};

class EmmaExporter : public FSExporter {
public:
    void exportTimeseries(const Label& title, const Label& xaxis, const Label& yaxis, const std::vector<TimeSeries>& data) const override; 
};

#endif
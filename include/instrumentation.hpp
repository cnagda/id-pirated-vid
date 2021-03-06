#ifndef INSTRUMENTATION_HPP
#define INSTRUMENTATION_HPP
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "database_iface.hpp"
#include "fs_compat.hpp"
#include <fstream>
#include <unordered_map>

namespace fs = std::filesystem;

struct MatchInfo
{
    std::string video;
    double confidence;
    double knownFrameRate;
    size_t startQuery, endQuery, startKnown, endKnown;
    double meanSimilarity;

    MatchInfo() = default;

    MatchInfo(std::string v, double c, double kfr, size_t sq, size_t eq, size_t sk, size_t ek, double ms = 0) : video(v), confidence(c), knownFrameRate(kfr), startQuery(sq), endQuery(eq), startKnown(sk), endKnown(eq), meanSimilarity(ms){
    }
};

struct FrameSimilarityInfo
{
    double similarity;
    IVideo::size_type f1Idx, f2Idx;
    const std::optional<IVideo> v1;
    const std::optional<IVideo> v2;
};

typedef std::string Label;

struct Point2f
{
    float x, y;
};

struct TimeSeries
{
    Label name;
    std::vector<Point2f> data;
};

class IExporter
{
public:
    virtual void exportTimeseries(const Label &title, const Label &xaxis, const Label &yaxis, const std::vector<TimeSeries> &data) const = 0;
};

class VideoMatchingInstrumenter
{
private:
    IVideo target;
    std::unordered_map<std::string, std::vector<Point2f>> videoTracker;

public:
    VideoMatchingInstrumenter(const IVideo &targetVideo) : target(targetVideo) {}
    void clear();
    void addFrameSimilarity(const FrameSimilarityInfo& info);
    std::vector<TimeSeries> getTimeSeries() const;
};

class FSExporter
{
protected:
    const fs::path outputDir;

public:
    FSExporter(fs::path dir = fs::current_path() / "Temp") : outputDir(dir){};
};

class CSVExporter : public FSExporter
{
    template<typename It>
    void writeRow(std::ostream& output, It begin, It end) const {
        if(begin != end) {
            output << *begin;
        }

        for(auto i = begin + 1; i != end; i++) {
            output << "," << *i;
        }

        output << std::endl;
    }
    template<typename It>
    void writeCSVToDir(const std::string& filename, const std::vector<std::string>& headers, It start, It end) const {
        std::ofstream output(outputDir / filename);

        auto n_cols = headers.size();
        writeRow(output, headers.begin(), headers.end());

        auto current = start;
        while(current != end) {
            auto row_end = current + n_cols;
            writeRow(output, current, row_end);
            current = row_end;
        }
    }
public:
    using FSExporter::FSExporter;
    const std::string delimiter = ",";
    void exportMatchLogs(const std::string& filename, const std::vector<MatchInfo>&) const;
};

class EmmaExporter : public FSExporter
{
public:
    void exportTimeseries(const Label &title, const Label &xaxis, const Label &yaxis, const std::vector<TimeSeries> &data) const;
};

#endif

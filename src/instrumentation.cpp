#include "instrumentation.hpp"
#include "fs_compat.hpp"
#include <fstream>
#include <algorithm>

void EmmaExporter::exportTimeseries(const Label &title, const Label &xaxis, const Label &yaxis, const std::vector<TimeSeries> &data) const
{
    fs::path search_dir(outputDir / title);
    fs::create_directories(search_dir);

    for (auto t : data)
    {
        std::ofstream out(search_dir / t.name);

        for (auto point : t.data)
        {
            out << point.y << std::endl;
        }
    }
}

void VideoMatchingInstrumenter::addFrameSimilarity(const FrameSimilarityInfo& info)
{
    if (info.v1 && info.v2)
    {
        auto& known = (info.v1->name == target.name) ? *info.v2 : *info.v1;
        videoTracker[known.name].push_back({static_cast<float>(info.f1Idx), static_cast<float>(info.similarity)});
    }
}

std::vector<TimeSeries> VideoMatchingInstrumenter::getTimeSeries() const
{
    std::vector<TimeSeries> out;
    for (auto &[name, points] : videoTracker)
    {
        std::vector<Point2f> dataCopy(points);
        std::sort(dataCopy.begin(), dataCopy.end(), [](auto p1, auto p2) { return p1.x < p2.x; });
        out.push_back({name, dataCopy});
    }

    return out;
}

void CSVExporter::exportMatchLogs(const std::string& filename, const std::vector<MatchInfo>& matches) const {
    std::vector<std::string> headers = {"Database Video", "Confidence", "Start Time", "End Time", "Query Start Time", "Query End Time"};
    std::vector<std::string> cells;
    for(const MatchInfo& match: matches) {
        cells.emplace_back(match.video);
        cells.emplace_back(std::to_string(match.confidence));
        cells.emplace_back(std::to_string(match.startKnown));
        cells.emplace_back(std::to_string(match.endKnown));
        cells.emplace_back(std::to_string(match.startQuery));
        cells.emplace_back(std::to_string(match.endQuery));
    }

    writeCSVToDir(filename, headers, cells.begin(), cells.end());
}
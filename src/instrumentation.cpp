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
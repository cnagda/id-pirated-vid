#include "instrumentation.hpp"
#include <experimental/filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::experimental::filesystem;

SimilarityReporter getReporter(VideoMatchingInstrumenter& instrumenter) {
    return [&instrumenter](auto f) { instrumenter.addFrameSimilarity(f); };
}

void VideoMatchingInstrumenter::addFrameSimilarity(FrameSimilarityInfo info) {
    if(info.v1 && info.v2) {
        auto& v1_ = *info.v1;
        auto& v2_ = *info.v2;
        auto& known = (v1_.name == target.name) ? v2_ : v1_ ;

        videoTracker[known.name].push_back({info.f1Idx, info.similarity}); 
    }
}

std::vector<TimeSeries> VideoMatchingInstrumenter::getTimeSeries() const {
    std::vector<TimeSeries> out;
    for(auto& [name, points] : videoTracker) {
        std::vector<Point2f> dataCopy(points);
        std::sort(dataCopy.begin(), dataCopy.end(), [](Point2f p1, Point2f p2){ return p1.x < p2.x; });
        out.push_back({name, dataCopy});
    }

    return out;
}

void EmmaExporter::exportTimeseries(Label title, Label xaxis, Label yaxis, const std::vector<TimeSeries>& series) {
    fs::path search_dir(outputDir / title);
    fs::create_directories(search_dir);

    for(auto t : series) {
        std::ofstream out(search_dir / t.name);

        for(auto point : t.data) {
            out << point.y << std::endl;
        }
    }
}
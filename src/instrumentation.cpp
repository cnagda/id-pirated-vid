#include "instrumentation.hpp"
#include <experimental/filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::experimental::filesystem;

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
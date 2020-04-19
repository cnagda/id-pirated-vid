#include "instrumentation.hpp"
#include <experimental/filesystem>
#include <fstream>

namespace fs = std::experimental::filesystem;

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
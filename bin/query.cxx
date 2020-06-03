#include <iostream>
#include <fstream>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "fs_compat.hpp"

#define DBPATH 1
#define VIDPATH 2

using namespace std;

bool file_exists(const string &fname)
{
    return fs::exists(fname);
}

int main(int argc, char **argv)
{

    bool DEBUG = false;
    if (argc < 3)
    {
        // TODO: better args parsing
        printf("usage: ./query <Database_Path> <test_video> --frames \n");
        return -1;
    }

    if (!fs::create_directories(fs::current_path() / "results"))
    {
        // std::cerr << "Could not create ./results" << std::endl;
    }

    std::ofstream f(fs::current_path() / "results" / "resultcache.txt", ios::out | ios::trunc);
    if (!f.is_open())
    {
        std::cerr << "Could not open ./results/resultcache.txt" << std::endl;
    }

    auto video = getSIFTVideo(argv[VIDPATH]);
    auto &fd = *query_database_factory(argv[DBPATH], -1, -1, -1).release();

    std::vector<MatchInfo> matches;

    if(argc == 4) {
        matches = findMatch(video.frames(), fd);
    } else if(argc == 3) {
        auto video2 = make_query_adapter(video, fd);
        matches = findMatch(video2, fd);
    }

    double queryFrameRate = video.getProperties().frameRate;

    std::vector<MatchInfo> timestampMatches(matches.size());
    std::transform(matches.begin(), matches.end(), timestampMatches.begin(), [queryFrameRate](auto match){
        return MatchInfo{
            match.video,
            match.confidence,
            match.knownFrameRate,
            static_cast<size_t>(match.startQuery / queryFrameRate * 1000.0),
            static_cast<size_t>(match.endQuery / queryFrameRate * 1000.0),
            static_cast<size_t>(match.startKnown / match.knownFrameRate * 1000.0),
            static_cast<size_t>(match.endKnown / match.knownFrameRate * 1000.0),
            match.meanSimilarity
        };
    });

    CSVExporter{"./results"}.exportMatchLogs(video.name + ".csv", timestampMatches);

    return 0;
}

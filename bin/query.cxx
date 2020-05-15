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
    auto video2 = make_query_adapter(video, fd);

    std::vector<MatchInfo> matches;

    if(argc == 4) {
        matches = findMatch(video.frames(), fd);
    } else if(argc == 3) {
        matches = findMatch(video2, fd);
    }

    std::string bestmatch = "";

    if(matches.size() > 0) {
        bestmatch = matches[0].video;
    }

    int count = 0;

    for(auto a: matches)
    {
        count++;
        std::cout << "Alignment " << count << ", Score: " << a.confidence << std::endl;
        std::cout << "Frame range in " << a.video << ": [" << a.startKnown << ", " << a.endKnown << ")" << std::endl;
        std::cout << "Frame range in " << video2.name << ": [" << a.startQuery << ", " << a.endQuery << ")" << std::endl
                    << std::endl;
    }

    f << bestmatch;
    f.close();

    return 0;
}

#include <iostream>
#include <string>
// #include <memory>
#include <vector>
#include <iomanip>
// #include "opencv2/highgui.hpp"
// #include "vocabulary.hpp"
#include "database.hpp"
// #include "matcher.hpp"
// #include "imgproc.hpp"
// #include "scene_detector.hpp"

#define DBPATH 1

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: ./info dbPath\n");
        return -1;
    }

    auto &db = *database_factory(argv[DBPATH], -1, -1, -1).release();

    auto config = db.getConfig();

    std::cout << "\nDatabase Configuration:" << std::endl;
    std::cout << std::left << std::setw(20) << "kFrame:" << config.KFrames << std::endl;
    std::cout << std::left << std::setw(20) << "kScene:" << config.KScenes << std::endl;
    std::cout << std::left << std::setw(20) << "thresholdScene:" << config.threshold << std::endl;

    auto vidList = db.listVideos();

    std::cout << "\nVideo List:" << std::endl;
    int count = 0;
    for (auto vid : vidList) {
        std::cout << std::left << std::setw(40) << vid << "\t";
        if (++count %2 == 0) {  std::cout << std::endl;  }
    }

}

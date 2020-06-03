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

#define WHITE       "\u001b[37m"
#define BLUE        "\033[94m"
#define GREEN       "\033[92m"
#define YELLOW      "\033[93m"
#define RED         "\033[91m"
#define ENDC        "\033[0m"
#define GREY        "\u001b[38;5;244m"
#define BOLD        "\u001b[1m"
#define UNDERLINE   "\u001b[4m"

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printf("usage: ./info dbPath\n");
        return -1;
    }

    auto &db = *database_factory(argv[DBPATH], -1, -1, -1).release();

    auto config = db.getConfig();

    std::cout << BOLD << "\nDatabase Configuration:" << ENDC << std::endl;
    std::cout << GREY << std::left << std::setw(20) << "kFrame:" << config.KFrames << ENDC << std::endl;
    std::cout << GREY << std::left << std::setw(20) << "kScene:" << config.KScenes << ENDC << std::endl;
    std::cout << GREY << std::left << std::setw(20) << "thresholdScene:" << config.threshold << ENDC << std::endl;

    auto vidList = db.listVideos();

    std::cout << BOLD << "\nVideo List:" << ENDC << std::endl;
    int count = 0;
    std::cout << GREY;
    for (auto vid : vidList) {
        std::cout << std::left << std::setw(40) << vid << "\t";
        if (++count %2 == 0) {  std::cout << std::endl;  }
    }
    std::cout << ENDC << std::endl;
}

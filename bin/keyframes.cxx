#include "vocabulary.hpp"
#include "matcher.hpp"
#include "scene_detector.hpp"
#include "database.hpp"
#include "subset.hpp"
#include <chrono>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

bool file_exists(const string &fname)
{
    return fs::exists(fname);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("usage: ./keyframes <Database_Path>\n");
        return -1;
    }

    namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.

    auto &fd = *database_factory(argv[1], -1, -1, -1).release();
    auto myvocab = loadVocabulary<Vocab<Frame>>(fd);
    // key frame stuff
    std::cout << "Key frame stuff" << std::endl;
    auto comp = BOWComparator(myvocab->descriptors());

    auto videopaths = fd.listVideos();
    std::cout << "Got video path list" << std::endl;

    for (auto video : videopaths)
    {
        auto vid = fd.loadVideo(video);
        auto ss = flatScenes(*vid, comp, .15);
        std::cout << "Video: " << vid->name << ", scenes: " << ss.size() << std::endl;
        for (auto &a : ss)
        {
            std::cout << a.first << ", ";
        }
        std::cout << std::endl;

        visualizeSubset(vid->name, ss.begin(), ss.end());
    }
    return 0;
}

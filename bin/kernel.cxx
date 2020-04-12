#include "kernel.hpp"
#include "database.hpp"
#include <raft>

#define THRESHOLD 3
#define DBPATH 1

using namespace std;

namespace fs = std::experimental::filesystem;

int isUnspecified(std::string arg) {
    return (arg == "-1");
}

int main(int argc, char** argv )
{
    if ( argc < 4 )
    {
        printf("usage: ./add dbPath vidPath threshScene \n");
        return -1;
    }

    double threshold = stod(argv[THRESHOLD]);

    auto db = database_factory(argv[DBPATH], -1, -1, threshold);

    raft::map m;
    VideoFrameSource source;
    ScaleImage scale;
    ExtractSIFT sift;
    ExtractColorHistogram color;
    DetectScene detect;
    ExtractFrame frame;
    ExtractScene scene;
    CollectFrame collect;
    SaveFrame saveFrame;
    SaveScene saveScene;

    m += source >> scale;
    m += scale >> color >> detect;
    m += scale >> sift >> frame;
    m += detect >> scene["scene_range"];
    m += frame >> scene["frame_descriptor"];
    m += scene >> saveScene;
    m += frame >> saveFrame;

    m.exe();

    return 0;
}

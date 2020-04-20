#include "kernel.hpp"
#include "database.hpp"
#include "vocabulary.hpp"
#include <raft>
#include <optional>
#include <memory>

#define DBPATH 1
#define VIDPATH 2
#define THRESHOLD 3 // TODO: do something with this??????

using namespace std;

namespace fs = std::experimental::filesystem;

int isUnspecified(std::string arg)
{
    return (arg == "-1");
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("usage: ./add dbPath vidPath threshScene\n");
        return -1;
    }

    double threshold = stod(argv[THRESHOLD]);

    auto db = database_factory(argv[DBPATH], -1, -1, threshold);
    std::string videoName(fs::path(argv[VIDPATH]).filename());

    auto frameVocab = loadVocabulary<Vocab<Frame>>(*db);
    auto sceneVocab = loadVocabulary<Vocab<SerializableScene>>(*db);

    raft::map m;
    VideoFrameSource source(argv[VIDPATH]);
    ScaleImage scale({600, 700});
    Duplicate<ordered_mat> scaledup, framedup, siftdup;
    Null<ordered_mat> siftDescriptor;
    ExtractSIFT sift;
    ExtractColorHistogram color;
    DetectScene detect(threshold);
    auto frame = frameVocab ? make_unique<ExtractFrame>(frameVocab.value()) : nullptr;
    auto scene = sceneVocab ? make_unique<ExtractScene>(sceneVocab.value()) : nullptr;
    SaveColor saveColor(videoName, db->getFileLoader());
    SaveFrameDescriptor saveFrame(videoName, db->getFileLoader());
    SaveSIFT saveSIFT(videoName, db->getFileLoader());
    SaveScene saveScene(videoName, db->getFileLoader());

    m += source <= scale >= scaledup;
    m += scaledup["first"] <= color >= saveColor;
    if (frame)
    {
        m += scaledup["second"] <= sift >= siftdup["in"];
        m += siftdup["first"] <= *frame >= saveFrame;
        m += siftdup["second"] >> saveSIFT;
    }
    else
    {
        m += scaledup["second"] <= sift >= siftDescriptor;
    }

    m.exe();

    return 0;
}

#include "filter.hpp"
#include "database.hpp"
#include "vocabulary.hpp"
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

    VideoFrameSource source(argv[VIDPATH]);
    ScaleImage scale({600, 700});
    ExtractSIFT sift;
    ExtractColorHistogram color;
    auto frame = frameVocab ? make_unique<ExtractFrame>(frameVocab.value()) : nullptr;
    SaveFrameSink saveFrame(videoName, db->getFileLoader());

    tbb::parallel_pipeline(300,
        tbb::make_filter<void, ordered_mat>(tbb::filter::serial_out_of_order, [&](tbb::flow_control& fc){
            return source(fc);
        }) &
        tbb::make_filter<ordered_mat, ordered_mat>(tbb::filter::parallel, scale) &
        tbb::make_filter<ordered_mat, std::pair<Frame, ordered_mat>>(tbb::filter::parallel, [&](auto mat){
            Frame f;
            f.descriptors = sift(mat).data;
            return make_pair(f, mat);
        }) &
        tbb::make_filter<std::pair<Frame, ordered_mat>, ordered_frame>(tbb::filter::parallel, [&](auto pair) {
            pair.first.colorHistogram = color(pair.second).data;
            return ordered_frame{pair.second.rank, pair.first};
        }) &
        tbb::make_filter<ordered_frame, void>(tbb::filter::parallel, saveFrame) 
    );

    return 0;
}

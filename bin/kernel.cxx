#include "filter.hpp"
#include "database.hpp"
#include "vocabulary.hpp"
#include <optional>
#include <memory>
#include <tbb/pipeline.h>
#include <opencv2/videoio.hpp>

#define DBPATH 1
#define VIDPATH 2
#define THRESHOLD 3 // TODO: do something with this??????

using namespace std;

int isUnspecified(std::string arg)
{
    return (arg == "-1");
}

class VideoFrameSource {
    cv::VideoCapture cap;
    size_t counter = 0;

public:
    VideoFrameSource(const std::string& path) : cap(path, cv::CAP_ANY) {}
    ordered_umat operator()(tbb::flow_control& fc) {
        cv::UMat image;
        if(cap.read(image)) {
            return {counter++, image};
        }
        fc.stop();
        return {};
    }
};

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

    auto frameVocab = loadVocabulary<Frame>(*db);
    auto sceneVocab = loadVocabulary<SerializableScene>(*db);

    VideoFrameSource source(argv[VIDPATH]);
    ScaleImage scale({600, 700});
    ExtractSIFT sift;
    ExtractColorHistogram color;
    auto frame = frameVocab ? make_unique<ExtractFrame>(frameVocab.value()) : nullptr;
    SaveFrameSink saveFrame(videoName, db->getFileLoader());

    tbb::parallel_pipeline(300,
        tbb::make_filter<void, ordered_umat>(tbb::filter::serial_out_of_order, [&](tbb::flow_control& fc){
            return source(fc);
        }) &
        tbb::make_filter<ordered_umat, ordered_umat>(tbb::filter::parallel, scale) &
        tbb::make_filter<ordered_umat, std::pair<Frame, ordered_umat>>(tbb::filter::parallel, [&](auto mat){
            Frame f;
            f.descriptors = sift(mat).data;
            return make_pair(f, mat);
        }) &
        tbb::make_filter<std::pair<Frame, ordered_umat>, ordered_frame>(tbb::filter::parallel, [&](auto pair) {
            pair.first.colorHistogram = color(pair.second).data;
            return ordered_frame{pair.second.rank, pair.first};
        }) &
        tbb::make_filter<ordered_frame, void>(tbb::filter::parallel, saveFrame) 
    );

    return 0;
}

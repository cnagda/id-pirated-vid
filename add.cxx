#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>
#include "opencv2/highgui.hpp"
#include "vocabulary.hpp"
#include "database.hpp"
#include "matcher.hpp"
#include "imgproc.hpp"
#include <future>

#define DBPATH      1
#define VIDPATH     2
#define KSCENE      3
#define KFRAME      4
#define THRESHOLD   5       // TODO: do something with this??????

using namespace cv;
using namespace std;

namespace fs = std::experimental::filesystem;

int isUnspecified(std::string arg) {
    return (arg == "-1");
}

int main(int argc, char** argv )
{
    bool DEBUG = false;
    if ( argc < 6 )
    {
        printf("usage: ./add dbPath vidPath kScene kFrame threshScene (DEBUG)\n");
        return -1;
    }
    if ( argc == 7 ) {
        DEBUG = true;
    }

    int kFrame = stoi(argv[KFRAME]);
    int kScene = stoi(argv[KSCENE]);
    double threshold = stod(argv[THRESHOLD]);

    namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.

    auto db = database_factory(argv[DBPATH], kFrame, kScene, threshold);

    if (!isUnspecified(argv[VIDPATH])) {
        auto video = make_video_adapter(getSIFTVideo(argv[VIDPATH], [DEBUG](UMat image, Frame frame){
            Mat output;
            auto descriptors = frame.descriptors;
            auto keyPoints = frame.keyPoints;

            if(DEBUG) {
                drawKeypoints(image, keyPoints, output);
                cout << "size: " << output.total() << endl;
                imshow("Display window", output);

                waitKey(0);

                auto im2 = scaleToTarget(image, 500, 700);
                imshow("Display window", im2);

                waitKey(0);
            }
        }), fs::path(argv[VIDPATH]).filename());

        db->saveVideo(video);
    }

    bool shouldRecalculateFrames = false;

    if(!isUnspecified(argv[KFRAME])) {
        auto v = constructFrameVocabulary(*db, kFrame, 10);
        saveVocabulary(v, *db);
        shouldRecalculateFrames = true;
    }
    if(!isUnspecified(argv[KSCENE])) {
        auto v = constructSceneVocabulary(*db, kScene);
        saveVocabulary(v, *db);
        shouldRecalculateFrames = true;
    }
    if(!isUnspecified(argv[THRESHOLD])) {
        shouldRecalculateFrames = true;
    }

    if(shouldRecalculateFrames) {
        auto vocab = loadVocabulary<Vocab<Frame>>(*db);
        auto sceneVocab = loadVocabulary<Vocab<SerializableScene>>(*db);

        if(!vocab) {
            throw std::runtime_error("no frame vocab");
        }

        for(auto entry : fs::directory_iterator(argv[DBPATH])) {
            if(fs::is_directory(entry) && fs::exists(entry.path() / "scenes")) {
                fs::remove_all(entry.path() / "scenes");
            }
        }

        std::shared_ptr db_shared(std::move(db));

        auto func = [fvocab = vocab->descriptors(), &sceneVocab](auto v, auto db) -> void {
            auto video = db->loadVideo(v);
            // try {
                for(Frame& frame : video->frames()) {
                    frame.frameDescriptor = getFrameDescriptor(frame, fvocab);
                }

                auto& scenes = video->getScenes();

                if(sceneVocab) {
                    for(SerializableScene& scene : scenes) {
                        scene.frameBag = getSceneDescriptor(scene, *video, *db);
                    }
                }

                db->saveVideo(*video);
            } catch(...) {
                std::cerr << "not enough info to compute scenes" << std::endl;
            }
        };

        std::vector<std::future<void>> runners;

        for(auto v : db_shared->listVideos()) {
            runners.push_back(std::async(std::launch::async, func, v, db_shared));
        }

        for(auto& i : runners) {
            i.wait();
        }
    }

    return 0;
}

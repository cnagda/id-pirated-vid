#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "opencv2/highgui.hpp"
#include "vocabulary.hpp"
#include "database.hpp"
#include "matcher.hpp"
#include "imgproc.hpp"
#include "scene_detector.hpp"

#define DBPATH 1
#define VIDPATH 2
#define KSCENE 3
#define KFRAME 4
#define THRESHOLD 5 // TODO: do something with this??????

using namespace cv;
using namespace std;

int isUnspecified(std::string arg)
{
    return (arg == "-1");
}

int main(int argc, char **argv)
{
    bool DEBUG = false;
    if (argc < 6)
    {
        printf("usage: ./add dbPath vidPath kScene kFrame threshScene (DEBUG)\n");
        return -1;
    }
    if (argc == 7)
    {
        DEBUG = true;
    }

    int kFrame = stoi(argv[KFRAME]);
    int kScene = stoi(argv[KSCENE]);
    double threshold = stod(argv[THRESHOLD]);

    if (DEBUG)
    {
        namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
    }

    auto db = database_factory(argv[DBPATH], kFrame, kScene, threshold);

    if (!isUnspecified(argv[VIDPATH]))
    {
        auto video = getSIFTVideo(argv[VIDPATH], [DEBUG](UMat image, Frame frame) {
            Mat output;
            auto descriptors = frame.descriptors;
            auto keyPoints = frame.keyPoints;

            if (DEBUG)
            {
                drawKeypoints(image, keyPoints, output);
                cout << "size: " << output.total() << endl;
                imshow("Display window", output);

                waitKey(0);

                auto im2 = scaleToTarget(image, 500, 700);
                imshow("Display window", im2);

                waitKey(0);
            }
        });

        auto saved = db->saveVideo(video);
        // get distances
        auto distances = get_distances(make_frame_source(db->getFileLoader(), saved->name, ColorHistogram), ColorComparator2D{});

        // std::cout << "distances size: " << distances.size() << std::endl;

        // make graph
        TimeSeries data;
        data.name = "data";
        for (int i = 0; i < distances.size(); i++)
        {
            data.data.push_back({0, static_cast<float>(distances[i])});
        }
        EmmaExporter().exportTimeseries(saved->name + "_timeseries", "Frame number", "Distance", {data});
        // get scenes
        auto scenes = hierarchicalScenes(distances, 30);
        // std::cout << saved->name << " scenes: " << std::endl;
        // for (auto &a : scenes)
        // {
        //     std::cout << a.first << ", " << a.second << std::endl;
        // }
    }

    bool shouldRecalculateFrames = false;

    if (!isUnspecified(argv[KFRAME]))
    {
        //auto v = constructFrameVocabulary(*db, kFrame, 10);
        // construct frame vocabulary running kmeans with max size 50000 to handle huge databases
        auto v = constructFrameVocabularyHierarchical(*db, kFrame, 50000, 10);
        saveVocabulary(v, *db);
        shouldRecalculateFrames = true;
    }
    if (!isUnspecified(argv[KSCENE]))
    {
        auto v = constructSceneVocabulary(*db, kScene);
        saveVocabulary(v, *db);
        shouldRecalculateFrames = true;
    }
    if (!isUnspecified(argv[THRESHOLD]))
    {
        shouldRecalculateFrames = true;
    }

    if (shouldRecalculateFrames)
    {
        for (auto& v : db->listVideos())
        {
            std::cout << std::endl << "Recalculating Scenes for " << v;
            auto video = db->loadVideo(v);
            db->saveVideo(*video);
        }
    }

    return 0;
}

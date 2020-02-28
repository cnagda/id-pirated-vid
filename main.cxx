#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"

#define DBPATH      1
#define VIDPATH     2
#define KSCENE      3
#define KFRAME      4
#define THRESHOLD   5       // TODO: do something with this??????
#define UNSPECIFIED "-1"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

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
    if(argv[KFRAME] == UNSPECIFIED) {
        kFrame = 200;
    }
    if(argv[KSCENE] == UNSPECIFIED) {
        kScene = 20;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.

    auto db = database_factory(argv[DBPATH], kFrame, kScene);

    if (argv[VIDPATH] != UNSPECIFIED) {
        auto video = make_video_adapter(getSIFTVideo(argv[VIDPATH], [DEBUG](Mat image, Frame frame){
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
        }), argv[VIDPATH]);

        db->saveVideo(video);
    }

    if(argv[KFRAME] != UNSPECIFIED) {
        auto v = constructFrameVocabulary(*db, kFrame);
        saveVocabulary(v, *db);
    }
    if(argv[KSCENE] != UNSPECIFIED) {
        auto v = constructSceneVocabulary(*db, kScene);
        saveVocabulary(v, *db);
    }

    return 0;
}

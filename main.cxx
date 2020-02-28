#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv )
{
    bool DEBUG = false;
    if ( argc < 2 )
    {
        printf("usage: ./main <Image_Path>\n");
        return -1;
    }
    if ( argc == 3 ) {
        DEBUG = true;
    }

    int KFrame = -1, KScene = -1;
    if(argc >= 3) {
        KFrame = 200;
    }
    if(argc >= 4) {
        KScene = 20;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.

    auto db = database_factory("database/", KFrame, KScene);
    auto video = make_video_adapter(getSIFTVideo(argv[1], [DEBUG](Mat image, Frame frame){
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
    }), argv[1]);

    db->saveVideo(video);
    if(KFrame > 0) {
        auto v = constructFrameVocabulary(*db, KFrame);
        saveVocabulary(v, *db);
    }
    if(KScene > 0) {
        auto v = constructSceneVocabulary(*db, KScene);
        saveVocabulary(v, *db);
    }

    return 0;
}

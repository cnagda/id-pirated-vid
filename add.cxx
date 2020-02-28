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
#define THRESHOLD   5
#define FRAMEVOCAB  "framevocab"
#define BOWPATH     "bowpath"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv )
{
    // ./add dbPath vidPath kScene kFrame threshScene (DEBUG)
    bool DEBUG = false;
    if ( argc < 6 )
    {
        printf("usage: ./add dbPath vidPath kScene kFrame threshScene (DEBUG)\n");
        return -1;
    }
    if ( argc == 7 ) {
        DEBUG = true;
    }

    namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.

    Mat myvocab;
    if (argv[KFRAME] != "-1" || !file_exists(BOWPATH)) {
        std::cout << "Calculating SIFT vocab" << std::endl;

        FileDatabase db(argv[1]);
        cv::Mat descriptors;
        for(auto &video : db.listVideos())
            for(auto &&frame : db.loadVideo(video)->frames())
                descriptors.push_back(frame.descriptors);

        Mat myvocab = constructVocabulary(descriptors, std::stoi(argv[KFRAME]));

        cv::FileStorage file(BOWPATH, cv::FileStorage::WRITE);

        file << "Vocabulary" << myvocab;
        file.release();
    } else {
        cv::FileStorage fs(BOWPATH, FileStorage::READ);

        fs["Vocabulary"] >> myvocab;
        fs.release();
    }

    Mat myframevocab;

    if (argv[KSCENE] != "-1" || !file_exists(FRAMEVOCAB)) {
        std::cout << "Calculating frame vocab" << std::endl;

        FileDatabase db(argv[1]);
        cv::Mat descriptors;
        for(auto &video : db.listVideos())
            for(auto &&frame : db.loadVideo(video)->frames())
                descriptors.push_back(baggify(frame.descriptors, myvocab));

        Mat myframeVocab = constructVocabulary(descriptors, std::stoi(argv[KSCENE]));

        cv::FileStorage file(FRAMEVOCAB, cv::FileStorage::WRITE);

        file << "Frame_Vocabulary" << myframeVocab;
        file.release();
    } else {
        cv::FileStorage fs2(FRAMEVOCAB, FileStorage::READ);

        fs2["Frame_Vocabulary"] >> myframevocab;
        fs2.release();
    }


    FileDatabase db(argv[DBPATH]);

    if (argv[VIDPATH] != "-1") {
        db.addVideo(argv[VIDPATH], [&](Mat image, Frame frame){
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
        });
    }

    return 0;
}

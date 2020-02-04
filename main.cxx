#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <sys/stat.h>
#include "database.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc < 2 )
    {
        printf("usage: ./main <Image_Path>\n");
        return -1;
    }

    namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

    FileDatabase db;
    db.addVideo(argv[1], [&](Mat image, Frame frame){
        Mat output;
        auto descriptors = frame.descriptors;
        auto keyPoints = frame.keyPoints;

        cout << descriptors.at<float>(37, 54) << endl;
        cout << descriptors.cols << endl;
        cout << descriptors.rows << endl;
        cout << descriptors.elemSize() << endl;
        cout << keyPoints.size() << endl << endl;

        drawKeypoints(image, keyPoints, output);
        cout << "size: " << output.total() << endl;
        imshow("Display window", output);

        waitKey(0);
    });
    return 0;
}

#include <stdio.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

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
    Mat image, output;

    namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

//     cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
// detector->detect( img, keypoints );
    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create();

    VideoCapture cap(argv[1], CAP_ANY);

    vector<KeyPoint> keypoints;

    while(cap.read(image)) {
    	detector->detect(image, keypoints);
    	drawKeypoints(image, keypoints, output);
        cout << "size: " << output.total() << endl;
        imshow("Display window", output);
    	waitKey(0);
    }
    return 0;
}

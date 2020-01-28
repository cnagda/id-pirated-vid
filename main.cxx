#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: ./main <Image_Path>\n");
        return -1;
    }
    Mat image;

    namedWindow("Display window", WINDOW_AUTOSIZE );// Create a window for display.

    SiftFeatureDetector detector;
    vector<cv::KeyPoint> keypoints;
    detector.detect(image, keypoints);

    // Add results to image and save.
    Mat output;


    VideoCapture cap(argv[1], CAP_ANY);
    while(cap.read(image)) {
	detector.detect(image, keypoints);
	drawKeypoints(image, keypoints, output);


        cout << "size: " << output.total() << endl;
        imshow("Display window", output);
	waitKey(0);
    }
    return 0;
}

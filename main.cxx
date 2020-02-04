#include <stdio.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void SIFTwrite(const string& filename, const Mat& mat, const vector<KeyPoint>& keyPoints);
pair<Mat, vector<KeyPoint>> SIFTread(const string& filename);

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
    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

    VideoCapture cap(argv[1], CAP_ANY);

    vector<KeyPoint> keyPoints;
    Mat descriptors;

    while(cap.read(image)) {
    	detector->detect(image, keyPoints);
        detector->compute(image, keyPoints, descriptors);

	/*for(int i = 0; i < descriptors.rows; i++){
		for(int j = 0; j < descriptors.cols; j++){
			cout << descriptors[descriptors.rows * i + j] << " ";	
		}
		cout << endl;
	}*/
	cout << descriptors.at<float>(37, 54) << endl;
	cout << descriptors.cols << endl;
	cout << descriptors.rows << endl;
	cout << descriptors.elemSize() << endl;
	cout << keyPoints.size() << endl << endl;

    	drawKeypoints(image, keyPoints, output);
        cout << "size: " << output.total() << endl;
        imshow("Display window", output);

        SIFTwrite("file.txt", descriptors, keyPoints);
	auto [testMat, testKP] = SIFTread("file.txt");

	cout << testMat.at<float>(37, 54) << endl;
	cout << testMat.cols << endl;
	cout << testMat.rows << endl;
	cout << testMat.elemSize() << endl;
	cout << testKP.size() << endl;

    	waitKey(0);
    }
    return 0;
}

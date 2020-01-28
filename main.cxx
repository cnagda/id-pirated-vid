#include <stdio.h>
#include <opencv2/opencv.hpp>

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


    VideoCapture cap(argv[1], CAP_ANY);
    while(cap.read(image)) {
        cout << "size: " << image.total() << endl;
        imshow("Display window", image);
    }
    return 0;
}
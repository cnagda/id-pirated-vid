#include <iostream>
#include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <sys/stat.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void SIFTwrite(const string& filename, const Mat& mat, const vector<KeyPoint>& keyPoints)
{
    ofstream fs(filename, fstream::binary);

    // Header
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char*)&mat.rows, sizeof(int));    // rows
    fs.write((char*)&mat.cols, sizeof(int));    // cols
    fs.write((char*)&type, sizeof(int));        // type
    fs.write((char*)&channels, sizeof(int));    // channels

    // Data
    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else
    {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r)
        {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }

    for(KeyPoint k : keyPoints) {
        fs << k.angle << k.class_id << k.octave << k.pt.x << k.pt.y << k.response << k.size;
    }
}

pair<Mat, vector<KeyPoint>> SIFTread(const string& filename)
{
    ifstream fs(filename, fstream::binary);

    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    Mat mat(rows, cols, type);
    vector<KeyPoint> keyPoints;
    for(int r = 0; r < rows; r++) {
        fs.read((char*)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    for(int r = 0; r < rows; r++) {
        KeyPoint k;
        fs >> k.angle >> k.class_id >> k.octave >> k.pt.x >> k.pt.y >> k.response >> k.size;
           keyPoints.push_back(k);
    }
    return make_pair(mat, keyPoints);
}

string getAlphas(string input){
    // TODO: check for at least one alpha char
    string output="";
    for (char const &c: input) {
		if ((c >= 65 && c <= 90) || (c >= 97 && c <= 122))
            output += c;
	}
    return output;
}

/* creates folder if it doesn't exist, otherwise breaks dramatically */
void createFolder(string folder_name) {
    // TODO: if no data folder, create it
    errno = 0;
    int res = mkdir(("./data/" + folder_name).c_str(), 0755);
    if (res != 0 && errno == EEXIST) {
        cout << "file with same name has been processed" << endl;
        exit(1);
    } else if (res != 0){
        cout << "Could not make directory" << endl;
        exit(1);
    }
}

int main(int argc, char** argv )
{
    if ( argc < 2 )
    {
        printf("usage: ./main <Image_Path>\n");
        return -1;
    }
    Mat image, output;

    namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

    VideoCapture cap(argv[1], CAP_ANY);

    vector<KeyPoint> keyPoints;
    Mat descriptors;

    // NOTE: what to do if overflow?
    unsigned long long int frame_num = 0;

    string folder_name = getAlphas(argv[1]);

    createFolder(folder_name);

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

        string file_name = folder_name + "_" + to_string(frame_num) + ".txt";
        SIFTwrite("./data/" + folder_name + "/" + file_name, descriptors, keyPoints);
        auto [testMat, testKP] = SIFTread("./data/" + folder_name + "/" + file_name);

        cout << testMat.at<float>(37, 54) << endl;
        cout << testMat.cols << endl;
        cout << testMat.rows << endl;
        cout << testMat.elemSize() << endl;
        cout << testKP.size() << endl;

        frame_num++;

        waitKey(0);
    }
    return 0;
}

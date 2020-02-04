#include "database.hpp"
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

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
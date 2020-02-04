#include "database.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <cctype>
#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

/* creates folder if it doesn't exist, otherwise throws an exception */
void createFolder(string folder_name) {
    fs::create_directories(fs::current_path() / folder_name);
}

string getAlphas(string input)
{
    // TODO: check for at least one alpha char
    string output;
    copy_if(input.begin(), input.end(), back_inserter(output), [](auto c) -> bool { return isalnum(c); });
    return output;
}

void SIFTwrite(const string &filename, const Mat &mat, const vector<KeyPoint> &keyPoints)
{
    ofstream fs(filename, fstream::binary);

    // Header
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char *)&mat.rows, sizeof(int)); // rows
    fs.write((char *)&mat.cols, sizeof(int)); // cols
    fs.write((char *)&type, sizeof(int));     // type
    fs.write((char *)&channels, sizeof(int)); // channels

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

    fs.write((char *)&keyPoints[0], keyPoints.size() * sizeof(KeyPoint));
}

pair<Mat, vector<KeyPoint>> SIFTread(const string &filename)
{
    ifstream fs(filename, fstream::binary);

    // Header
    int rows, cols, type, channels;
    fs.read((char *)&rows, sizeof(int));     // rows
    fs.read((char *)&cols, sizeof(int));     // cols
    fs.read((char *)&type, sizeof(int));     // type
    fs.read((char *)&channels, sizeof(int)); // channels

    // Data
    Mat mat(rows, cols, type);
    vector<KeyPoint> keyPoints;
    for (int r = 0; r < rows; r++)
    {
        fs.read((char *)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    for (int r = 0; r < rows; r++)
    {
        KeyPoint k;
        fs.read((char *)&k, sizeof(KeyPoint));
        keyPoints.push_back(k);
    }
    return make_pair(mat, keyPoints);
}

unique_ptr<IVideo> FileDatabase::addVideo(const std::string &filepath, std::function<void(Mat, Frame)> callback)
{
    fs::path video_dir = fs::current_path() / fs::path("database") / getAlphas(filepath);
    fs::create_directories(video_dir);
    vector<Frame> frames;

    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

    VideoCapture cap(filepath, CAP_ANY);

    vector<KeyPoint> keyPoints;
    Mat descriptors, image;

    size_t index = 0;

    while (cap.read(image))
    { // test only loading 2 frames
        detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);

        Frame frame{keyPoints, descriptors.clone()};

        frames.push_back(frame);

        SIFTwrite(video_dir / to_string(index++), descriptors, keyPoints);

        if(callback) callback(image, frame);
    }

    return make_unique<SIFTVideo>(frames);
}
unique_ptr<IVideo> FileDatabase::loadVideo(const std::string &filepath)
{
    fs::path video_dir = fs::current_path() / fs::path("database") / getAlphas(filepath);
    vector<Frame> frames;

    auto it = fs::directory_iterator{video_dir};
    vector<fs::directory_entry> files(it, fs::end(it));
    sort(files.begin(), files.end(), [](auto a, auto b) {
        return stoi(a.path().filename()) < stoi(b.path().filename());
    });

    for (auto frame_path : files)
    {
        auto [mat, keyPoints] = SIFTread(frame_path.path());
        frames.push_back(Frame{keyPoints, mat.clone()});
    }
    return make_unique<SIFTVideo>(frames);
}
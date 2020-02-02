#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <experimental/filesystem>
#include "database.hpp"

using namespace std;
using namespace cv;

namespace fs = std::experimental::filesystem::v1;

void SIFTwrite(const string& filename, const Mat& mat, const vector<KeyPoint>& keyPoints);
pair<Mat, vector<KeyPoint>> SIFTread(const string& filename);

class DatabaseAdaptor : public IDatabase {
public:
    unique_ptr<IVideo> addVideo(const std::string& filepath) {
        fs::path video_dir = fs::current_path() / fs::path("database") / filepath;
        fs::create_directories(video_dir);
        vector<Frame> frames;

        Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

        VideoCapture cap(filepath, CAP_ANY);

        vector<KeyPoint> keyPoints;
        Mat descriptors, image;

        size_t index = 0;

        while(cap.read(image) && index < 2) { // test only loading 2 frames
            detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);
            frames.push_back(Frame{keyPoints, descriptors});

            SIFTwrite(video_dir / to_string(index++), descriptors, keyPoints);
            break;
        }

        return make_unique<SIFTVideo>(frames);
    }
    unique_ptr<IVideo> loadVideo(const std::string& filepath) {
        fs::path video_dir = fs::current_path() / fs::path("database") / filepath;
        vector<Frame> frames;

        auto it = fs::directory_iterator{video_dir};
        vector<fs::directory_entry> files(it, fs::end(it));
        sort(files.begin(), files.end(), [](auto a, auto b){
            return stoi(a.path().filename()) < stoi(b.path().filename());
        });

        for(auto frame_path : files) {
            auto [mat, keyPoints] = SIFTread(frame_path.path());
            frames.push_back(Frame{keyPoints, mat});
        }
        return make_unique<SIFTVideo>(frames);
    }
};

bool videoLoadSaveTest(IDatabase& database) {
    auto result = database.addVideo("sample.mp4")->frames();
    auto loaded = database.loadVideo("sample.mp4")->frames();

    return equal(result.begin(), result.end(), loaded.begin());
}

TEST(DatabaseSuite, InstantiateTest) {
    DatabaseAdaptor db;
    ASSERT_TRUE(videoLoadSaveTest(db));
}
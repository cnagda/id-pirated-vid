#include "gtest/gtest.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <experimental/filesystem>
#include "database.hpp"

using namespace std;
using namespace cv;
namespace fs = experimental::filesystem;

TEST(DatabaseSuite, getAlphas) {
    string input = "this is a{} good";
    ASSERT_TRUE(getAlphas(input) == "thisisagood");
}

TEST(DatabaseSuite, SIFTrwTest) {
    Mat mat1({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    Mat mat2({3, 2}, {10, 11, 12, 13, 14, 15});

    vector<KeyPoint> k1{KeyPoint(0.1f, 0.2f, 0.3f), KeyPoint(0.15f, 0.25f, 0.35f)};
    vector<KeyPoint> k2{KeyPoint(0.2f, 0.3f, 0.4f), KeyPoint(0.25f, 0.35f, 0.45f), KeyPoint(0.4f, 0.5f, 0.6f)};

    vector<Frame>result {Frame{k1, mat1}, Frame{k2, mat2}};
    
    SIFTwrite("test_frame1", result[0]);
    SIFTwrite("test_frame2", result[1]);

    auto f1 = SIFTread("test_frame1");
    auto f2 = SIFTread("test_frame2");
    vector<Frame> loaded{f1, f2};

    ASSERT_TRUE(result.size() > 0);
    ASSERT_TRUE(result.size() == loaded.size());
    ASSERT_TRUE(equal(result.begin(), result.end(), loaded.begin()));
}

TEST(DatabaseSuite, FileDatabase) {
    FileDatabase db;
    auto vid = db.addVideo("sample.mp4")->frames();
    auto loaded = db.loadVideo("sample.mp4")->frames();

    cout << "size: " << vid.size() << endl;

    ASSERT_TRUE(vid.size() > 0);
    ASSERT_TRUE(vid.size() == loaded.size());
    ASSERT_TRUE(equal(vid.begin(), vid.end(), loaded.begin()));
}
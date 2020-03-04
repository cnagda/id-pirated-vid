#include "gtest/gtest.h"
#include <iostream>
#include <experimental/filesystem>
#include "database.hpp"

using namespace std;
using namespace cv;
namespace fs = experimental::filesystem;

#ifdef CLEAN_NAMES
TEST(DatabaseSuite, getAlphas) {
    string input = "this is a{} good";
    EXPECT_TRUE(getAlphas(input) == "thisisagood");
}
#endif

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

    EXPECT_TRUE(result.size() > 0);
    EXPECT_TRUE(result.size() == loaded.size());
    EXPECT_TRUE(equal(result.begin(), result.end(), loaded.begin()));
}

TEST(DatabaseSuite, FileDatabase) {
    FileDatabase db(std::make_unique<LazyStorageStrategy>(), RuntimeArguments{200, 20});
    auto video = make_video_adapter(getSIFTVideo("../sample.mp4"), "sample.mp4");
    
    auto vid = db.saveVideo(video)->frames();
    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto& loaded = loaded_ptr->frames();
    cout << "size: " << vid.size() << endl;

    EXPECT_TRUE(vid.size() > 0);
    EXPECT_TRUE(vid.size() == loaded.size());
    EXPECT_TRUE(equal(vid.begin(), vid.end(), loaded.begin()));
}

TEST(DatabaseSuite, EagerDatabase) {
    FileDatabase db(std::make_unique<AggressiveStorageStrategy>(), RuntimeArguments{200, 20});
    auto video = make_video_adapter(getSIFTVideo("../sample.mp4"), "sample.mp4");
    
    auto vid = db.saveVideo(video);
    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto& loaded = loaded_ptr->frames();
    cout << "size: " << vid->frames().size() << endl;

    EXPECT_TRUE(vid->frames().size() > 0);
    EXPECT_TRUE(vid->frames().size() == loaded.size());
    EXPECT_TRUE(equal(vid->frames().begin(), vid->frames().end(), loaded.begin()));

    EXPECT_TRUE(vid->getScenes().size() > 0);
    EXPECT_TRUE(vid->getScenes().size() == loaded_ptr->getScenes().size());
    // EXPECT_TRUE(equal(vid->getScenes().begin(), vid->getScenes().end(), loaded_ptr->getScenes().begin()));
}
#include "gtest/gtest.h"
#include <iostream>
#include <experimental/filesystem>
#include "database.hpp"
#include "vocabulary.hpp"

using namespace std;
using namespace cv;
namespace fs = experimental::filesystem;

#ifdef CLEAN_NAMES
TEST(DatabaseSuite, getAlphas) {
    string input = "this is a{} good";
    EXPECT_TRUE(getAlphas(input) == "thisisagood");
}
#endif


class DatabaseSuite : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        input = getSIFTVideo("../sample.mp4");
    }
    void SetUp() override {
        try {
            fs::remove_all(fs::current_path() / "database_test_dir");
        } catch(std::exception e) {
            std::cout << "could not remove test dir" << e.what() << std::endl;
        }

        fs::create_directories(fs::current_path() / "database_test_dir");
    }

    static SIFTVideo input;
};

SIFTVideo DatabaseSuite::input;

TEST(FileRW, SIFTrwTest) {
    Mat mat1({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    Mat mat2({3, 2}, {10, 11, 12, 13, 14, 15});
    Mat mat3({1, 2}, {5, 4});
    Mat mat4({2, 2}, {4, 4, 4, 4});
    const int levels = 20;
    const cv::Size size = cv::Size(123, 234);

    const cv::Mat proto = cv::Mat(size, CV_16SC1, 1000);

    std::vector<cv::Mat> channels;
    for (int i = 0; i < levels; i++)
        channels.push_back(proto);

    cv::Mat A;
    cv::merge(channels, A);

    vector<KeyPoint> k1{KeyPoint(0.1f, 0.2f, 0.3f), KeyPoint(0.15f, 0.25f, 0.35f)};
    vector<KeyPoint> k2{KeyPoint(0.2f, 0.3f, 0.4f), KeyPoint(0.25f, 0.35f, 0.45f), KeyPoint(0.4f, 0.5f, 0.6f)};

    vector<Frame>result {Frame{k1, mat1, mat3, A}, Frame{k2, mat2, mat4, A}};

    SIFTwrite("test_frame1", result[0]);
    SIFTwrite("test_frame2", result[1]);

    auto f1 = SIFTread("test_frame1");
    auto f2 = SIFTread("test_frame2");
    vector<Frame> loaded{f1, f2};

    EXPECT_TRUE(result.size() > 0);
    EXPECT_TRUE(result.size() == loaded.size());
    EXPECT_TRUE(equal(result.begin(), result.end(), loaded.begin()));
}

TEST_F(DatabaseSuite, FileDatabase) {
    FileDatabase db(fs::current_path() / "database_test_dir",
        std::make_unique<LazyStorageStrategy>(),
        LazyLoadStrategy{},
        RuntimeArguments{200, 20, 0.2});
    auto video = make_video_adapter(input, "sample.mp4");

    auto vid = db.saveVideo(video)->frames();
    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto& loaded = loaded_ptr->frames();
    cout << "size: " << vid.size() << endl;

    EXPECT_TRUE(vid.size() > 0);
    EXPECT_TRUE(vid.size() == loaded.size());
    EXPECT_TRUE(equal(vid.begin(), vid.end(), loaded.begin()));
}

TEST_F(DatabaseSuite, EagerDatabase) {
    FileDatabase db(fs::current_path() / "database_test_dir",
        std::make_unique<AggressiveStorageStrategy>(),
        AggressiveLoadStrategy{},
        RuntimeArguments{200, 20, 0.2});

    auto in = make_video_adapter(input, "sample.mp4");
    auto in_saved = db.saveVideo(in);
    saveVocabulary(constructFrameVocabulary(db, db.getConfig().KFrames, 10), db);

    auto vid = db.saveVideo(*in_saved);
    EXPECT_GT(vid->getScenes().size(), 0);
    EXPECT_TRUE(vid->getScenes()[0].frameBag.empty());

    saveVocabulary(constructSceneVocabulary(db, db.getConfig().KScenes), db);

    auto vid_3 = db.saveVideo(*vid);

    EXPECT_FALSE(vid_3->getScenes()[0].frameBag.empty());

    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto& loaded = loaded_ptr->frames();
    cout << "size: " << vid->frames().size() << endl;

    EXPECT_GT(vid->frames().size(), 0);
    EXPECT_EQ(vid->frames().size(), loaded.size());
    EXPECT_TRUE(equal(vid->frames().begin(), vid->frames().end(), loaded.begin()));

    EXPECT_GT(vid->getScenes().size(), 0);
    EXPECT_EQ(vid->getScenes().size(), loaded_ptr->getScenes().size());
    // EXPECT_TRUE(equal(vid->getScenes().begin(), vid->getScenes().end(), loaded_ptr->getScenes().begin()));
}

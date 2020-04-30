#include "gtest/gtest.h"
#include <iostream>
#include <experimental/filesystem>
#include "database.hpp"
#include "vocabulary.hpp"
#include "matcher.hpp"

using namespace std;
using namespace cv;
namespace fs = experimental::filesystem;

#ifdef CLEAN_NAMES
TEST(DatabaseSuite, getAlphas)
{
    string input = "this is a{} good";
    EXPECT_TRUE(getAlphas(input) == "thisisagood");
}
#endif

class DatabaseSuite : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        input = getSIFTVideo("../sample.mp4");
    }
    void SetUp() override
    {
        try
        {
            fs::remove_all(fs::current_path() / "database_test_dir");
        }
        catch (std::exception e)
        {
            std::cout << "could not remove test dir" << e.what() << std::endl;
        }

        fs::create_directories(fs::current_path() / "database_test_dir");
    }

    static SIFTVideo input;
};

SIFTVideo DatabaseSuite::input;

TEST(FileRW, SIFTrwTest)
{
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

    vector<Frame> result{Frame{k1, mat1, mat3, A}, Frame{k2, mat2, mat4, A}};

    SIFTwrite("test_frame1", result[0]);
    SIFTwrite("test_frame2", result[1]);

    auto f1 = SIFTread("test_frame1");
    auto f2 = SIFTread("test_frame2");
    vector<Frame> loaded{f1, f2};

    EXPECT_TRUE(result.size() > 0);
    EXPECT_TRUE(result.size() == loaded.size());
    EXPECT_TRUE(equal(result.begin(), result.end(), loaded.begin(), loaded.end()));
}

TEST(FileRW, FileLoaderRWTest)
{
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

    vector<Frame> result{Frame{k1, mat1, mat3, A}, Frame{k2, mat2, mat4, A}};

    if (fs::exists("loader_test"))
        fs::remove_all("loader_test");

    FileLoader loader("loader_test");
    loader.initVideoDir("test_video");
    loader.saveFrame("test_video", 0, result[0]);
    loader.saveFrame("test_video", 1, result[1]);

    auto f1 = loader.readFrame("test_video", 0);
    auto f2 = loader.readFrame("test_video", 1);
    vector<Frame> loaded{*f1, *f2};

    EXPECT_TRUE(result.size() > 0);
    EXPECT_TRUE(result.size() == loaded.size());
    EXPECT_TRUE(equal(result.begin(), result.end(), loaded.begin(), loaded.end()));
}

TEST_F(DatabaseSuite, FileDatabase)
{
    FileDatabase db(to_string(fs::current_path() / "database_test_dir"),
                    std::make_unique<LazyStorageStrategy>(),
                    LazyLoadStrategy{},
                    RuntimeArguments{200, 20, 0.2});

    auto vid = read_all(*db.saveVideo(input)->frames());
    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto loaded = read_all(*loaded_ptr->frames());
    cout << "size: " << vid.size() << endl;

    EXPECT_TRUE(vid.size() > 0);
    EXPECT_TRUE(vid.size() ==loaded.size());
    EXPECT_TRUE(equal(vid.begin(), vid.end(), loaded.begin()));
}

TEST_F(DatabaseSuite, EagerDatabase)
{
    FileDatabase db(to_string(fs::current_path() / "database_test_dir"),
                    std::make_unique<AggressiveStorageStrategy>(),
                    AggressiveLoadStrategy{},
                    RuntimeArguments{200, 20, 30});

    auto in_saved = db.saveVideo(input);
    saveVocabulary(constructFrameVocabulary(db, db.getConfig().KFrames, 10), db);

    auto vid = db.saveVideo(*in_saved);
    auto scenes = read_all(*vid->getScenes());
    ASSERT_GT(scenes.size(), 0);
    EXPECT_TRUE(scenes[0].frameBag.empty());

    saveVocabulary(constructSceneVocabulary(db, db.getConfig().KScenes), db);

    auto vid_3 = db.saveVideo(*vid);

    auto scene = vid_3->getScenes()->read();
    ASSERT_TRUE(scene);
    EXPECT_FALSE(scene->frameBag.empty());

    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto loaded = read_all(*loaded_ptr->frames());
    auto frames = read_all(*vid->frames());

    EXPECT_GT(frames.size(), 0);
    EXPECT_EQ(frames.size(), loaded.size());
    EXPECT_TRUE(equal(frames.begin(), frames.end(), loaded.begin()));

    auto loaded_scenes = read_all(*loaded_ptr->getScenes());

    EXPECT_TRUE(equal(scenes.begin(), scenes.end(), loaded_scenes.begin(), loaded_scenes.end(), [](auto& a, auto& b){
        return a.startIdx == b.startIdx && a.endIdx == b.endIdx;
    }));
}

TEST_F(DatabaseSuite, QueryAdapter)
{
    FileDatabase db(to_string(fs::current_path() / "database_test_dir"),
                    std::make_unique<AggressiveStorageStrategy>(),
                    AggressiveLoadStrategy{},
                    RuntimeArguments{200, 20, 30});

    auto in_saved = db.saveVideo(input);
    ASSERT_TRUE(in_saved);

    saveVocabulary(constructFrameVocabulary(db, db.getConfig().KFrames, 10), db);
    saveVocabulary(constructSceneVocabulary(db, db.getConfig().KScenes), db);

    auto loaded_ptr = db.loadVideo("sample.mp4");

    ASSERT_TRUE(loaded_ptr);

    auto loaded = read_all(*loaded_ptr->frames());
    ASSERT_GT(loaded.size(), 0);

    auto loaded_scenes = read_all(*loaded_ptr->getScenes());
    ASSERT_GT(loaded_scenes.size(), 0);

    auto vocab = loadVocabulary<Vocab<Frame>>(db);
    ASSERT_TRUE(vocab);

    auto original_frames = read_all(*input.frames(*vocab));

    EXPECT_TRUE(equal(original_frames.begin(), original_frames.end(), loaded.begin(), loaded.end(), [](auto &a, auto &b) {
        if(!matEqual(a.descriptors, b.descriptors)) {
            std::cout << "SIFT descriptors don't match" << std::endl;
            return false;
        }
        if(cosineSimilarity(a.frameDescriptor, b.frameDescriptor) < 0.95) {
            std::cout << a.frameDescriptor << std::endl;
            std::cout << b.frameDescriptor << std::endl;
            std::cout << "frame bag doesn't match" << std::endl;
            return false;
        }
        if(!matEqual(a.colorHistogram, b.colorHistogram)) {
            std::cout << "color histogram doesn't match" << std::endl;
            return false;
        }

        return true;
    }));

    auto original_scenes = read_all(*make_query_adapter(input, db).getScenes());
    EXPECT_TRUE(equal(original_scenes.begin(), original_scenes.end(), loaded_scenes.begin(), loaded_scenes.end(), [](auto &a, auto &b) {
        return (a.startIdx == b.startIdx && a.endIdx == b.endIdx && cosineSimilarity(a.frameBag, b.frameBag) > 0.95);
    }));
}
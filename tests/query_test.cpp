#include "gtest/gtest.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include "database.hpp"
#include "matcher.hpp"

namespace fs = std::experimental::filesystem;

class DatabaseFixture : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        fs::remove_all(fs::current_path() / "database_test_dir");
        db = std::make_unique<FileDatabase>(fs::current_path() / "database_test_dir", 
            std::make_unique<LazyStorageStrategy>(),
            RuntimeArguments{200, 20});
        {
            auto video = make_video_adapter(
                getSIFTVideo("../sample.mp4"), "sample.mp4");
            db->saveVideo(video);
        }

        {
            auto video = make_video_adapter(
                getSIFTVideo("../coffee.mp4"), "coffee.mp4");
            db->saveVideo(video);
        }

        {
            auto video = make_video_adapter(
                getSIFTVideo("../crab.mp4"), "crab.mp4");
            db->saveVideo(video);
        }

        saveVocabulary(constructFrameVocabulary(*db, 200), *db);
        saveVocabulary(constructSceneVocabulary(*db, 20), *db);
        
        std::cout << "Setup done" << std::endl;
    }

    static std::unique_ptr<FileDatabase> db;
};

std::unique_ptr<FileDatabase> DatabaseFixture::db;

TEST_F(DatabaseFixture, NotInDatabase) {  
    auto video = InputVideoAdapter<SIFTVideo>(getSIFTVideo("../sample.mp4"), "sample.mp4");
    auto match = findMatch(video, *db);
    if(match) {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    EXPECT_FALSE(match.has_value());
}

TEST_F(DatabaseFixture, InDatabase) {
    auto video = db->loadVideo()[0].release();
    EXPECT_TRUE(findMatch(*video, *db).has_value());
}

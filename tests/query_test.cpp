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
        try { 
            fs::remove_all(fs::current_path() / "database_test_dir");
        } catch(std::exception e) {
            std::cout << "could not remove test dir" << e.what() << std::endl;
        }

        db = std::make_unique<FileDatabase>(fs::current_path() / "database_test_dir", 
            std::make_unique<LazyStorageStrategy>(),
            LazyLoadStrategy{},
            RuntimeArguments{200, 20, 0.15});

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
    auto video = make_query_adapter(*db, getSIFTVideo("../sample.mp4"), "sample.mp4");
    auto match = findMatch(video, *db);
    if(match) {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    EXPECT_FALSE(match.has_value());
}

TEST_F(DatabaseFixture, InDatabase) {
    auto video = db->loadVideo(db->listVideos()[0]);
    auto match = findMatch(*video, *db);
    ASSERT_TRUE(match.has_value());
    auto topMatch = match.value();
    EXPECT_EQ(topMatch.video, video->name);
    EXPECT_EQ(topMatch.startFrame, 0);
    EXPECT_EQ(topMatch.endFrame, video->getScenes().size());
}

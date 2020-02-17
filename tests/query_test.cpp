#include "gtest/gtest.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include "database.hpp"
#include "matcher.hpp"

namespace fs = std::experimental::filesystem;

class DatabaseFixture : public ::testing::Test {
protected:
    void SetUp() override {
        fs::remove_all(fs::current_path() / "database_test");
        db = FileDatabase(fs::current_path() / "database_test");

        db.addVideo("../coffee.mp4");
        db.addVideo("../crab.mp4");

        vocab = constructVocabulary(fs::current_path() / "database_test", 200, 10);
    }

    FileDatabase db;
    cv::Mat vocab;
};

TEST_F(DatabaseFixture, NotInDatabase) {  
    auto video = getSIFTVideo("../sample.mp4");
    auto match = findMatch(video, db, vocab);
    if(match) {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    ASSERT_FALSE(match.has_value());
}

TEST_F(DatabaseFixture, InDatabase) {
    auto video = db.loadVideo(db.listVideos()[0]);
    ASSERT_TRUE(findMatch(*video, db, vocab).has_value());
}

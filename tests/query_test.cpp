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
        db = FileDatabase(fs::current_path() / "database_test_dir");

        db.addVideo("../coffee.mp4");
        db.addVideo("../crab.mp4");

        cv::Mat descriptors;
        for(auto &video : db.listVideos()) {
            auto frames = db.loadVideo(video)->frames();
            for(auto &&frame : frames)
                descriptors.push_back(frame.descriptors);
        }

        vocab = constructVocabulary(descriptors, 200);
        
        descriptors = cv::Mat();
        for(auto &video : db.listVideos()) {
            auto frames = db.loadVideo(video)->frames();
            for(auto &&frame : frames)
                descriptors.push_back(baggify(frame.descriptors, vocab));
        }

        frameVocab = constructVocabulary(descriptors, 20);

        std::cout << "Setup done" << std::endl;
    }

    static FileDatabase db;
    static cv::Mat vocab, frameVocab;
};

FileDatabase DatabaseFixture::db;
cv::Mat DatabaseFixture::vocab;
cv::Mat DatabaseFixture::frameVocab;

TEST_F(DatabaseFixture, NotInDatabase) {  
    auto video = getSIFTVideo("../sample.mp4");
    auto match = findMatch(video, db, vocab, frameVocab);
    if(match) {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    EXPECT_FALSE(match.has_value());
}

TEST_F(DatabaseFixture, InDatabase) {
    auto video = db.loadVideo(db.listVideos()[0]);
    EXPECT_TRUE(findMatch(*video, db, vocab, frameVocab).has_value());
}

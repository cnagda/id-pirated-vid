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
        SubdirSearchStrategy strat("../");
        EagerStorageStrategy store;

        {
            SIFTVideo v = strat("sample.mp4", [](auto s) { return getSIFTVideo(s); });
            DatabaseVideo<decltype(v)> video(v);
            store.saveVideo(video, db);
        }

        {
            SIFTVideo v = strat("coffee.mp4", [](auto s) { return getSIFTVideo(s); });
            DatabaseVideo<decltype(v)> video(v);
            store.saveVideo(video, db);
        }

        {
            SIFTVideo v = strat("crab.mp4", [](auto s) { return getSIFTVideo(s); });
            DatabaseVideo<decltype(v)> video(v);
            store.saveVideo(video, db);
        }

        static_cast<IDatabase&>(db).saveVocab(constructFrameVocabulary(db, 2000));
        static_cast<IDatabase&>(db).saveVocab(constructSceneVocabulary(db, 200));
        
        std::cout << "Setup done" << std::endl;
    }

    static FileDatabase db;
};

FileDatabase DatabaseFixture::db;

TEST_F(DatabaseFixture, NotInDatabase) {  
    auto video = DatabaseVideo<SIFTVideo>(getSIFTVideo("../sample.mp4"));
    auto match = findMatch(video, db);
    if(match) {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    EXPECT_FALSE(match.has_value());
}

TEST_F(DatabaseFixture, InDatabase) {
    auto video = db.loadVideo()[0].release();
    EXPECT_TRUE(findMatch(*video, db).has_value());
}

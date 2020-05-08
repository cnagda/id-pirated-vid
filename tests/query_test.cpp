#include "gtest/gtest.h"
#include <iostream>
#include "database.hpp"
#include "matcher.hpp"
#include "vocabulary.hpp"

class DatabaseFixture : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        try
        {
            fs::remove_all(fs::current_path() / "database_test_dir");
        }
        catch (std::exception e)
        {
            std::cout << "could not remove test dir" << e.what() << std::endl;
        }

        db = std::make_unique<FileDatabase>(to_string(fs::current_path() / "database_test_dir"),
                                            std::make_unique<AggressiveStorageStrategy>(),
                                            LazyLoadStrategy{},
                                            RuntimeArguments{200, 20, 30});

        {
            db->saveVideo(getSIFTVideo("../coffee.mp4"));
        }

        {
            db->saveVideo(getSIFTVideo("../crab.mp4"));
        }

        saveVocabulary(constructFrameVocabulary(*db, 200), *db);
        saveVocabulary(constructSceneVocabulary(*db, 20), *db);

        for (auto video : db->listVideos())
        {
            auto vid = db->loadVideo(video);
            db->saveVideo(*vid);
        }

        std::cout << "Setup done" << std::endl;
    }

    static std::unique_ptr<FileDatabase> db;
};

std::unique_ptr<FileDatabase> DatabaseFixture::db;

TEST_F(DatabaseFixture, NotInDatabase)
{
    auto video = make_query_adapter(getSIFTVideo("../sample.mp4"), *db);
    auto match = findMatch(video, *db);
    if (match)
    {
        std::cout << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl;
    }
    EXPECT_LT(match->matchConfidence, 6);
}

TEST_F(DatabaseFixture, InDatabase)
{
    auto video = db->loadVideo(db->listVideos()[0]);
    ASSERT_TRUE(video);
    auto match = findMatch(make_query_adapter(*video), *db);
    ASSERT_TRUE(match);
    auto topMatch = match.value();
    auto scenes = read_all(*video->getScenes());
    EXPECT_EQ(topMatch.video, video->name);
    EXPECT_EQ(topMatch.startFrame, 0);
    EXPECT_EQ(topMatch.endFrame, scenes.size());
}


TEST_F(DatabaseFixture, InDatabaseQueryAdapter)
{
    auto video = make_query_adapter(getSIFTVideo("../coffee.mp4"), *db);
    auto match = findMatch(video, *db);
    ASSERT_TRUE(match);
    auto topMatch = match.value();
    auto scenes = read_all(*db->loadVideo("coffee.mp4")->getScenes());
    EXPECT_EQ(topMatch.video, video.name);
    EXPECT_EQ(topMatch.startFrame, 0);
    EXPECT_EQ(topMatch.endFrame, scenes.size());
}

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
    auto matches = findMatch(video, *db);
    for(auto match: matches)
    {
        std::cout << "confidence: " << match.confidence << " video: " << match.video << std::endl;
        EXPECT_LT(match.confidence, 6);
    }
}

TEST_F(DatabaseFixture, InDatabase)
{
    auto video = db->loadVideo(db->listVideos()[0]);
    ASSERT_TRUE(video);

    auto matches = findMatch(make_query_adapter(*video), *db);

    ASSERT_GT(matches.size(), 0);
    auto topMatch = matches[0];

    EXPECT_EQ(topMatch.video, video->name);
    EXPECT_EQ(topMatch.startKnown, 0);
    EXPECT_EQ(topMatch.startKnown, topMatch.startQuery);
    EXPECT_EQ(topMatch.endKnown, topMatch.endQuery);
    EXPECT_EQ(topMatch.endKnown, video->loadMetadata().frameCount);
}


TEST_F(DatabaseFixture, InDatabaseQueryAdapter)
{
    auto inputVideo = getSIFTVideo("../coffee.mp4");

    auto matches = findMatch(make_query_adapter(inputVideo, *db), *db);

    ASSERT_GT(matches.size(), 0);
    auto topMatch = matches[0];

    EXPECT_EQ(topMatch.video, inputVideo.name);
    EXPECT_EQ(topMatch.startKnown, 0);
    EXPECT_EQ(topMatch.startKnown, topMatch.startQuery);
    EXPECT_EQ(topMatch.endKnown, topMatch.endQuery);
    EXPECT_EQ(topMatch.endKnown, inputVideo.getProperties().frameCount);
}

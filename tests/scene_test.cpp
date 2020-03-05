#include "scene.hpp"
#include "gtest/gtest.h"

TEST(SceneTest, Is_copiable) {
    cv::Mat data{5, 6, 7, 7, 8};
    SerializableScene s1{data, 5, 7};
    SerializableScene s2(s1);
    EXPECT_EQ(s1.startIdx, s2.startIdx);
    EXPECT_EQ(s1.endIdx, s2.endIdx);
    EXPECT_TRUE(matEqual(s1.frameBag, s2.frameBag));
}
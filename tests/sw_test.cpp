#include "sw.hpp"
#include "gtest/gtest.h"
#include <algorithm>

TEST(SmithWatterman_Suite, Equality_Test) {
    std::string s1 = "hello";
    auto results = calculateAlignment(s1.begin(), s1.end(), s1.begin(), s1.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 3, 2);
    ASSERT_TRUE(results.size() > 0);

    EXPECT_TRUE(std::distance(results[0].startKnown, results[0].endKnown) == s1.size());
    EXPECT_TRUE(std::distance(results[0].startUnknown, results[0].endUnknown) == s1.size());
    EXPECT_EQ(results[0].startKnown, s1.begin());
    EXPECT_EQ(results[0].endKnown, s1.end());
    EXPECT_EQ(results[0].startUnknown, s1.begin());
    EXPECT_EQ(results[0].endUnknown, s1.end());

    std::string s2 = "owagesdgnjbsuhellonwamsdjhby";
    results = calculateAlignment(s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 3, 2);
    ASSERT_TRUE(results.size() > 0);
    auto first = results[0];

    EXPECT_TRUE(std::distance(first.startKnown, first.endKnown) == s1.size());
    EXPECT_TRUE(std::distance(first.startUnknown, first.endUnknown) == s1.size());
    EXPECT_TRUE(std::equal(first.startKnown, first.endKnown, first.startUnknown, first.endUnknown));
}


TEST(SmithWatterman_Suite, No_Match) {
    std::string s1 = "hello";
    std::string s2 = "tatris";
    auto results = calculateAlignment(s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 3, 2);
    ASSERT_TRUE(results.size() == 0);
}
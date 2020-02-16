#include "sw.hpp"
#include "gtest/gtest.h"
#include <algorithm>

TEST(SmithWatterman_Suite, Equality_Test) {
    std::string s1 = "hello";
    auto results = calculateAlignment(s1.begin(), s1.end(), s1.begin(), s1.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 10, 2);
    ASSERT_TRUE(results.size() > 0);

    ASSERT_TRUE(std::distance(results[0].startKnown, results[0].endKnown) == s1.size());
    ASSERT_TRUE(std::distance(results[0].startUnknown, results[0].endUnknown) == s1.size());
    ASSERT_EQ(results[0].startKnown, s1.begin());
    ASSERT_EQ(results[0].endKnown, s1.end());
    ASSERT_EQ(results[0].startUnknown, s1.begin());
    ASSERT_EQ(results[0].endUnknown, s1.end());

    std::string s2 = "owagesdgnjbsuhellonwamsdjhby";
    results = calculateAlignment(s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 10, 2);
    auto first = results[0];

    ASSERT_TRUE(std::distance(first.startKnown, first.endKnown) == s1.size());
    ASSERT_TRUE(std::distance(first.startUnknown, first.endUnknown) == s1.size());
    ASSERT_TRUE(std::equal(first.startKnown, first.endKnown, first.startUnknown, first.endUnknown));
}


TEST(SmithWatterman_Suite, No_Match) {
    std::string s1 = "hello";
    std::string s2 = "tatris";
    auto results = calculateAlignment(s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 10, 2);
    ASSERT_TRUE(results.size() == 0);
}
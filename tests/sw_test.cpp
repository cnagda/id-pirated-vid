#include "sw.hpp"
#include "gtest/gtest.h"

TEST(SmithWatterman_Suite, Equality_Test) {
    std::string s1 = "hello";
    auto results = calculateAlignment(s1.begin(), s1.end(), s1.begin(), s1.end(), [](auto a, auto b){ return a == b ? 3 : -3; }, 10, 2);
    ASSERT_TRUE(results.size() > 0);
    ASSERT_EQ(results[0].startKnown, s1.begin());
    ASSERT_EQ(results[0].endKnown, s1.end());
    ASSERT_EQ(results[0].startUnknown, s1.begin());
    ASSERT_EQ(results[0].endUnknown, s1.end());
}
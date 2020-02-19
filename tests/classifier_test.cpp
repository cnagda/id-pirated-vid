#include "gtest/gtest.h"
#include "classifier.hpp"

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

template<typename... Args> bool alwaysTrueClassifier(Args&&...) { return true; }

TEST(Classifier, getAccuracy) {
    std::vector<labelled_test<int>> tests;
    std::generate_n(std::back_inserter(tests), 100, [i = 0, val = false]() mutable {
        auto ret = std::make_pair(val, i++);
        val = !val;
        return ret;
    });
    
    {
        auto&& results = BinaryResults::runClassifier(tests.begin(), tests.end(), alwaysTrueClassifier<int>);
        auto&& matrix = results.getConfusionMatrix();
        EXPECT_EQ(0.5, getAccuracy(matrix));
    }

    std::generate_n(tests.begin(), 100, []{ return std::make_pair(false, 0); });

    {
        auto&& results = BinaryResults::runClassifier(tests.begin(), tests.end(), alwaysTrueClassifier<int>);
        EXPECT_EQ(0, getAccuracy(results.getConfusionMatrix()));
    }

    std::generate_n(tests.begin(), 100, []{ return std::make_pair(true, 0); });

    {
        auto&& results = BinaryResults::runClassifier(tests.begin(), tests.end(), alwaysTrueClassifier<int>);
        EXPECT_EQ(1, getAccuracy(results.getConfusionMatrix()));
    }
}
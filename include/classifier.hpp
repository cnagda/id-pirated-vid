#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP
#include <utility>
#include <vector>
#include <cmath>

/* given a classifier, a labelled set of good values, and a labelled set of bad values
 * run the classifier on the tests and collect the results */

template<typename T> using labelled_test = std::pair<bool, T>;

template<typename T> struct ConfusionMatrix {
    T tp, tn, fp, fn;
};

class BinaryResults {
// pair label, guess
typedef std::pair<bool, bool> classification_results;
const std::vector<classification_results> results;
typedef decltype(results)::size_type size_type;

BinaryResults(std::vector<classification_results>&& results) : results(results) {}
BinaryResults(const std::vector<classification_results>& results) : results(results) {}

public:
    ConfusionMatrix<size_type> getConfusionMatrix() const {
        // zero initialize with brackets
        ConfusionMatrix<size_type> r{};
        for(auto& i : results) {
            r.tp += static_cast<bool>(i.first && i.second);
            r.tn += static_cast<bool>(!i.first && !i.second);
            r.fp += static_cast<bool>(!i.first && i.second);
            r.fn += static_cast<bool>(i.first && !i.second);
        }

        return r;
    }
    // It -> first the class of the test
    // It -> second the data of the test
    template<typename It, typename C> static BinaryResults runClassifier(It testBegin, It testEnd, C&& c) {
        std::vector<classification_results> results;
        for(auto i = testBegin; i != testEnd; i++) 
            results.push_back(
                std::make_pair(i->first, std::invoke(c, std::forward<decltype(i->second)>(i->second))));

        return {results};
    }
};

template <typename T> double getAccuracy(const ConfusionMatrix<T>& m) {
    return static_cast<double>(m.tp + m.tn) / static_cast<double>(m.tp + m.fn + m.fp + m.tn);
}

template <typename T> double getPrecision(const ConfusionMatrix<T>& m) {
    return static_cast<double>(m.tp) / static_cast<double>(m.tp + m.fp);
}

template <typename T> double getSensitivity(const ConfusionMatrix<T>& m) {
    return static_cast<double>(m.tp) / static_cast<double>(m.tp + m.fn);
}

template <typename T> double getSpecificity(const ConfusionMatrix<T>& m) {
    return static_cast<double>(m.tn) / static_cast<double>(m.tn + m.fp);
}

template <typename T> double getRecall(const ConfusionMatrix<T>& m) {
    return static_cast<double>(m.tp) / static_cast<double>(m.tp + m.tn);
}

template <typename T> double getFMeasure(const ConfusionMatrix<T>& m) {
    auto& p = getPrecision(m);
    auto& r = getRecall(m);
    return 2 * p * r / (p + r);
}

template <typename T> double getGeometricMean(const ConfusionMatrix<T>& m) {
    return std::sqrt(m.tp * m.tn);
}

#endif
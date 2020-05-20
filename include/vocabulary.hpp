#ifndef KEYFRAMES_HPP
#define KEYFRAMES_HPP

#include "concepts.hpp"
#include <iostream>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <type_traits>
#include "vocab_type.hpp"
#include "video.hpp"

class FileDatabase;

template <typename Matrix>
cv::Mat constructVocabulary(Matrix &&descriptors, unsigned int K, cv::Mat labels = cv::Mat())
{
    //cv::BOWKMeansTrainer trainer(K);
    cv::Mat retval;

    cv::kmeans(descriptors, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, retval);

    // std::cout << "About to return" << std::endl;

    return retval;
    //return trainer.cluster(descriptors);
}

template <typename It>
cv::Mat constructVocabulary(It start, It end, unsigned int K, cv::Mat labels = cv::Mat())
{
    cv::Mat accumulator;
    for (auto i = start; i != end; ++i)
        accumulator.push_back(*i);

    cv::UMat copy;
    accumulator.copyTo(copy);

    return constructVocabulary(copy, K, labels);
}

Vocab<Frame> constructFrameVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator = 1);
Vocab<SerializableScene> constructSceneVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator = 1);
Vocab<Frame> constructFrameVocabularyHierarchical(const FileDatabase &database, unsigned int K, unsigned int N, unsigned int speedinator);
Vocab<SerializableScene> constructSceneVocabularyHierarchical(const FileDatabase &database, unsigned int K, unsigned int N, unsigned int speedinator);

class BOWExtractor {
    cv::BOWImgDescriptorExtractor extractor;

public:
    BOWExtractor(const BOWExtractor& w) : extractor(cv::FlannBasedMatcher::create()) {
        extractor.setVocabulary(w.extractor.getVocabulary());
    }
    BOWExtractor(const ContainerVocab& vocab) : extractor(cv::FlannBasedMatcher::create()) {
        extractor.setVocabulary(vocab.descriptors());
    }
    BOWExtractor(const cv::Mat& mat) : extractor(cv::FlannBasedMatcher::create()) {
        extractor.setVocabulary(mat);
    }

    template<typename Input, typename Output>
    auto compute(Input&& input, Output&& output) {
        return extractor.compute(std::forward<Input>(input), std::forward<Output>(output));
    }
};

template <typename Matrix, typename Extractor>
cv::Mat baggify(Matrix &&f, Extractor &&extractor)
{
    cv::Mat output;

    if (!f.empty())
    {
        extractor.compute(f, output);
    }
    else
    {
        // std::cerr << "In baggify: Frame dimension does not match vocab" << std::endl;
    }

    return output;
}

template <typename It, typename Extractor>
cv::Mat baggify(It rangeBegin, It rangeEnd, Extractor &&extractor)
{
    cv::Mat accumulator;
    for (auto i = rangeBegin; i != rangeEnd; ++i)
        accumulator.push_back(*i);
    return baggify(accumulator, std::forward<Extractor>(extractor));
}

template <typename It, typename Extractor>
inline cv::Mat baggify(std::pair<It, It> pair, Extractor &&extractor)
{
    return baggify(pair.first, pair.second, std::forward<Extractor>(extractor));
}

template <typename V, typename Db>
bool saveVocabulary(V &&vocab, Db &&db)
{
    return db.saveVocab(std::forward<V>(vocab), std::remove_reference_t<V>::vocab_name);
}

template <typename T, typename Db>
std::optional<Vocab<T>> loadVocabulary(Db &&db)
{
    auto v = db.loadVocab(T::vocab_name);
    if (v)
    {
        return Vocab<T>(v.value());
    }
    return std::nullopt;
}

template <typename T, typename Db>
std::optional<Vocab<T>> loadOrComputeVocab(Db &&db, int K)
{
    auto vocab = loadVocabulary<T>(std::forward<Db>(db));
    if (!vocab)
    {
        if (K == -1)
        {
            return std::nullopt;
        }

        Vocab<T> v;
        if constexpr (std::is_same_v<T, Frame>)
        {
            v = constructFrameVocabulary(db, K, 10);
        }
        else if constexpr (std::is_base_of_v<T, SerializableScene>)
        {
            v = constructSceneVocabulary(db, K);
        }
        saveVocabulary(v, std::forward<Db>(db));
        return v;
    }
    return vocab.value();
}

// Tries to read cached value of frame descriptor, or else will build and cache it
template <class Extractor>
cv::Mat loadFrameDescriptor(Frame &frame, Extractor &&extractor)
{
    if (frame.frameDescriptor.empty())
    {
        frame.frameDescriptor = getFrameDescriptor(frame, std::forward<Extractor>(extractor));
    }
    return frame.frameDescriptor;
}

// Same as above, but does not save to cache
template <class Extractor>
cv::Mat loadFrameDescriptor(const Frame &frame, Extractor &&extractor)
{
    if (frame.frameDescriptor.empty())
    {
        return getFrameDescriptor(frame, std::forward<Extractor>(extractor));
    }
    return frame.frameDescriptor;
}

// get a descriptor from frame's sift data
template <class Extractor>
inline cv::Mat getFrameDescriptor(const Frame &frame, Extractor &&extractor)
{
    return baggify(frame.descriptors, std::forward<Extractor>(extractor));
}

class BOWComparator
{
    BOWExtractor extractor;

public:
    template <typename Vocab>
    BOWComparator(Vocab&& vocab) : extractor(std::forward<Vocab>(vocab)){};

    double operator()(Frame &f1, Frame &f2) const;
    double operator()(Frame &f1, Frame &f2);
};

#endif

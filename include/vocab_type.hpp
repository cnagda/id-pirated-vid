#ifndef VOCAB_TYPE_HPP
#define VOCAB_TYPE_HPP
#include <string>
#include <opencv2/core/mat.hpp>

class ContainerVocab {
private:
    cv::Mat desc;
public:
    ContainerVocab() = default;
    ContainerVocab(const cv::Mat& descriptors) : desc(descriptors) {};

    cv::Mat descriptors() const {
        return desc;
    }
};

template<typename T>
class Vocab : public ContainerVocab {
public:
    using ContainerVocab::ContainerVocab;
    Vocab(const ContainerVocab& v) : ContainerVocab(v) {};
    static const char* const vocab_name;
};

template <typename T>
const char* const Vocab<T>::vocab_name = T::vocab_name;

#endif
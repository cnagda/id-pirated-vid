#ifndef VOCAB_TYPE_HPP
#define VOCAB_TYPE_HPP
#include <string>
#include <opencv2/core/mat.hpp>

class ContainerVocab {
private:
    cv::Mat desc;
    std::string hash;
public:
    ContainerVocab() = default;
    ContainerVocab(const cv::Mat& descriptors) : desc(descriptors), hash() {};
    std::string getHash() const {
        return hash;
    }
    cv::Mat descriptors() const {
        return desc;
    }
    static const std::string vocab_name;
};

template<typename T>
class Vocab : public ContainerVocab {
public:
    using ContainerVocab::ContainerVocab;
    Vocab(const ContainerVocab& v) : ContainerVocab(v) {};
    static const std::string vocab_name;
    typedef T vocab_type;
};

#endif
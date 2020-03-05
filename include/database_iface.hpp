#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "frame.hpp"
#include <exception>

class FileDatabase;
class SerializableScene;

class IScene {
public:
    static const std::string vocab_name;

    virtual ~IScene() = default;

    virtual const cv::Mat& descriptor() = 0;
    virtual const std::vector<Frame>& getFrames() = 0;
    virtual operator SerializableScene() = 0;
};


class IVideo {
public:
    IVideo(const std::string& name) : name(name) {};
    IVideo(const IVideo& video) : name(video.name) {};
    IVideo(IVideo&& video) : name(video.name) {};
    const std::string name;
    using size_type = std::vector<Frame>::size_type;

    virtual size_type frameCount() = 0;
    virtual std::vector<Frame>& frames() & = 0;
    virtual std::vector<std::unique_ptr<IScene>>& getScenes() & = 0;
    virtual ~IVideo() = default;
};


template<typename T>
class ICursor {
public:
    virtual ICursor& advance() & = 0;
    virtual operator bool() = 0;
    virtual const T& getValue() const & = 0;
};

enum StrategyType {
    Lazy,
    Eager
};

class IVideoStorageStrategy {
public:
    virtual StrategyType getType() const = 0;
    virtual bool shouldBaggifyFrames(IVideo& video) = 0;
    virtual bool shouldComputeScenes(IVideo& video) = 0;
    virtual bool shouldBaggifyScenes(IVideo& video) = 0;
    virtual ~IVideoStorageStrategy() = default;
};

/* templated thing
class IVideoLoadStrategy {
public:
    static StrategyType getType() = 0;
    static bool shouldLoadFrames() = 0;
    static bool shouldLoadScenes() = 0;
}; 
*/


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
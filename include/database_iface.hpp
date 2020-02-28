#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "frame.hpp"
#include <exception>

class IScene {
public:
    static const std::string vocab_name;

    IScene(const std::string& key) : key(key) {};
    virtual ~IScene() = default;
    const std::string key;

    virtual cv::Mat descriptor() = 0;
    virtual std::vector<Frame>& getFrames() & = 0;
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

class IVideoStorageStrategy {
public:
    virtual bool shouldBaggifyFrames(IVideo& video) = 0;
    virtual bool shouldComputeScenes(IVideo& video) = 0;
    virtual bool shouldBaggifyScenes(IVideo& video) = 0;
    virtual ~IVideoStorageStrategy() = default;
};

#endif
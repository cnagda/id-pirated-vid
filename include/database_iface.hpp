#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <vector>

struct Frame;
struct SerializableScene;
class IVideo {
public:
    IVideo(const std::string& name) : name(name) {};
    IVideo(const IVideo& video) : name(video.name) {};
    IVideo(IVideo&& video) : name(video.name) {};
    const std::string name;
    using size_type = std::vector<Frame>::size_type;

    virtual size_type frameCount() = 0;
    virtual std::vector<Frame>& frames() & = 0;
    virtual std::vector<SerializableScene>& getScenes() & = 0;
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

#endif
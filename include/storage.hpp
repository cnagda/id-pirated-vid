#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "database_iface.hpp"
#include <experimental/filesystem>
#include "video.hpp"

class FileLoader {
private:
    fs::path rootDir;
public:
    explicit FileLoader(fs::path dir) : rootDir(dir) {};

    std::optional<Frame> readFrame(const std::string& videoName, v_size index) const;
    std::optional<SerializableScene> readScene(const std::string& videoName, v_size index) const;
    bool saveFrame(const std::string& videoName, v_size index, const Frame& frame);
    bool saveScene(const std::string& videoName, v_size index, const SerializableScene& scene);

    void initVideoDir(const std::string& videoName);
};



// TODO think of how to use these, templated
/* Example strategies
class IVideoLoadStrategy {
public:
    virtual std::unique_ptr<IVideo> operator()(const std::string& findKey) const = 0;
    virtual ~IVideoLoadStrategy() = default;
};

 */

class AggressiveStorageStrategy : public IVideoStorageStrategy {
public:
    inline StrategyType getType() const { return Eager; };
    inline bool shouldBaggifyFrames(IVideo& video) override { return true; };
    inline bool shouldComputeScenes(IVideo& video) override { return true; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return true; };
};

class LazyStorageStrategy : public IVideoStorageStrategy {
public:
    inline StrategyType getType() const { return Lazy; };
    inline bool shouldBaggifyFrames(IVideo& video) override { return false; };
    inline bool shouldComputeScenes(IVideo& video) override { return false; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return false; };
};

struct AggressiveLoadStrategy {
    constexpr operator StrategyType() { return Eager; };
};

struct LazyLoadStrategy {
    constexpr operator StrategyType() { return Lazy; };
};

#endif
#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "database_iface.hpp"
#include <experimental/filesystem>
#include "video.hpp"
#include <utility>

namespace fs = std::experimental::filesystem;

void clearDir(fs::path path);

template <class T, class RankType>
struct ordered_adapter
{
    RankType rank;
    T data;
    ordered_adapter() = default;
    ordered_adapter(const RankType &rank, const T &data) : rank(rank), data(data){};
    bool operator<(const ordered_adapter &a) const { return rank < a.rank; };
};

class FileLoader
{
private:
    fs::path rootDir;
    bool saveFrameData(const std::string &videoName, const std::string &fileName, const cv::Mat &mat) const;
    std::optional<cv::Mat> readFrameData(const std::string &videoName, const std::string &fileName) const;

public:
    FileLoader(const std::string& path) : rootDir(path) {};

    std::optional<Frame> readFrame(const std::string &videoName, size_t index) const;
    std::optional<cv::Mat> readFrameFeatures(const std::string &videoName, size_t index) const;
    std::optional<cv::Mat> readFrameColorHistogram(const std::string &videoName, size_t index) const;
    std::optional<cv::Mat> readFrameBag(const std::string &videoName, size_t index) const;

    std::optional<SerializableScene> readScene(const std::string &videoName, size_t index) const;

    bool saveFrame(const std::string &videoName, size_t index, const Frame &frame) const;
    bool saveFrameFeatures(const std::string &videoName, size_t index, const cv::Mat &mat) const;
    bool saveFrameColorHistogram(const std::string &videoName, size_t index, const cv::Mat &mat) const;
    bool saveFrameBag(const std::string &videoName, size_t index, const cv::Mat &mat) const;

    bool saveScene(const std::string &videoName, size_t index, const SerializableScene &scene) const;

    template <typename Range>
    bool saveRange(Range &&range, const std::string &video, size_t offset) const
    {
        return std::all_of(std::begin(range), std::end(range), [this, &video, &offset](const auto &element) mutable {
            if constexpr (std::is_convertible_v<decltype(element), SerializableScene>)
            {
                return saveScene(video, offset++, element);
            }
            else if constexpr (std::is_convertible_v<decltype(element), Frame>)
            {
                return saveFrame(video, offset++, element);
            }
        });
    }

    void initVideoDir(const std::string &videoName) const;
    void clearFrames(const std::string &videoName) const;
    void clearScenes(const std::string &videoName) const;
};

// TODO think of how to use these, templated
/* Example strategies
class IVideoLoadStrategy {
public:
    virtual std::unique_ptr<IVideo> operator()(const std::string& findKey) const = 0;
    virtual ~IVideoLoadStrategy() = default;
};

 */

class AggressiveStorageStrategy : public IVideoStorageStrategy
{
public:
    inline StrategyType getType() const override { return Eager; };
    inline bool shouldBaggifyFrames() const override { return true; };
    inline bool shouldComputeScenes() const override { return true; };
    inline bool shouldBaggifyScenes() const override { return true; };
};

class LazyStorageStrategy : public IVideoStorageStrategy
{
public:
inline StrategyType getType() const override { return Lazy; };
    inline bool shouldBaggifyFrames() const override { return false; };
    inline bool shouldComputeScenes() const override { return false; };
    inline bool shouldBaggifyScenes() const override { return false; };
};

struct AggressiveLoadStrategy
{
    constexpr operator StrategyType() { return Eager; };
};

struct LazyLoadStrategy
{
    constexpr operator StrategyType() { return Lazy; };
};

// converts a functor to the concept required by get_distances
template<typename F> struct read_adapter {
    std::decay_t<F> f;
    read_adapter(F&& f) : f(f) {}
    read_adapter(const F& f) : f(f) {}

    auto read() { return f(); }
};

auto inline make_frame_source(const FileLoader& loader, const std::string& videoName) {
    return read_adapter{[&loader, &videoName, index = 0]() mutable {
        return loader.readFrame(videoName, index++);
    }};
}

auto inline make_scene_source(const FileLoader& loader, const std::string& videoName) {
    return read_adapter{[&loader, &videoName, index = 0]() mutable {
        return loader.readScene(videoName, index++);
    }};
}

auto inline make_color_source(const FileLoader& loader, const std::string& videoName) {
    return read_adapter{[&loader, &videoName, index = 0]() mutable {
        return loader.readFrameColorHistogram(videoName, index++);
    }};
}

#endif
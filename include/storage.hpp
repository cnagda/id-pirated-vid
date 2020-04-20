#ifndef STORAGE_HPP
#define STORAGE_HPP

#include "database_iface.hpp"
#include <experimental/filesystem>
#include "video.hpp"

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

    std::optional<Frame> readFrame(const std::string &videoName, v_size index) const;
    std::optional<cv::Mat> readFrameFeatures(const std::string &videoName, v_size index) const;
    std::optional<cv::Mat> readFrameColorHistogram(const std::string &videoName, v_size index) const;
    std::optional<cv::Mat> readFrameBag(const std::string &videoName, v_size index) const;

    std::optional<SerializableScene> readScene(const std::string &videoName, v_size index) const;

    bool saveFrame(const std::string &videoName, v_size index, const Frame &frame) const;
    bool saveFrameFeatures(const std::string &videoName, v_size index, const cv::Mat &mat) const;
    bool saveFrameColorHistogram(const std::string &videoName, v_size index, const cv::Mat &mat) const;
    bool saveFrameBag(const std::string &videoName, v_size index, const cv::Mat &mat) const;

    bool saveScene(const std::string &videoName, v_size index, const SerializableScene &scene) const;

    template <typename Range>
    bool saveRange(Range &&range, const std::string &video, v_size offset) const
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
    inline StrategyType getType() const { return Eager; };
    inline bool shouldBaggifyFrames(IVideo &video) override { return true; };
    inline bool shouldComputeScenes(IVideo &video) override { return true; };
    inline bool shouldBaggifyScenes(IVideo &video) override { return true; };
};

class LazyStorageStrategy : public IVideoStorageStrategy
{
public:
    inline StrategyType getType() const { return Lazy; };
    inline bool shouldBaggifyFrames(IVideo &video) override { return false; };
    inline bool shouldComputeScenes(IVideo &video) override { return false; };
    inline bool shouldBaggifyScenes(IVideo &video) override { return false; };
};

struct AggressiveLoadStrategy
{
    constexpr operator StrategyType() { return Eager; };
};

struct LazyLoadStrategy
{
    constexpr operator StrategyType() { return Lazy; };
};

#endif
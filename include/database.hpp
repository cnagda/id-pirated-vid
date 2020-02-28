#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <functional>
#include <experimental/filesystem>
#include <type_traits>
#include <optional>
#include "database_iface.hpp"

namespace fs = std::experimental::filesystem;

class SIFTVideo;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);
SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});
cv::Mat scaleToTarget(cv::Mat image, int targetWidth, int targetHeight);

typedef std::string hash_default;

/*
template<typename T>
class LazyVocab : IVocab {
private:
    fs::path directory;
public:
    LazyVocab(fs::path directory) : directory(directory) {};
    Hash getHash() const override {
    }
    Matrix descriptors() const override {
        Matrix myvocab;
        cv::FileStorage fs(directory / T::vocab_name, cv::FileStorage::READ);
        fs["Vocabulary"] >> myvocab;
        return myvocab;
    }
    static const std::string vocab_name;
};

template<typename T>
const std::string LazyVocab<T>::vocab_name = T::vocab_name;
*/

class ContainerVocab : public IVocab {
private:
    const cv::Mat desc;
    std::string hash;
public:
    ContainerVocab(const cv::Mat& descriptors) : desc(descriptors) {};
    ContainerVocab(const IVocab& vocab) : desc(vocab.descriptors()) {};
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
    static const std::string vocab_name;
};

class SIFTVideo {
private:
    std::vector<Frame> SIFTFrames;
    using size_type = std::vector<Frame>::size_type;
public:
    SIFTVideo(const std::vector<Frame>& frames) : SIFTFrames(frames) {};
    SIFTVideo(std::vector<Frame>&& frames) : SIFTFrames(frames) {};
    SIFTVideo(SIFTVideo&& vid) : SIFTFrames(vid.SIFTFrames) {};
    SIFTVideo(const SIFTVideo& vid) : SIFTFrames(vid.SIFTFrames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

template<typename Base>
class InputVideoAdapter : public IVideo{
private:
    Base base;
    std::vector<std::unique_ptr<IScene>> emptyScenes;
public:
    InputVideoAdapter(Base&& b, const std::string& name) : IVideo(name), base(std::forward<Base>(b)) {};
    InputVideoAdapter(const Base& b, const std::string& name) : IVideo(name), base(std::forward<Base>(b)) {};
    InputVideoAdapter(IVideo&& vid) : IVideo(vid), base(vid.frames()) {};
    InputVideoAdapter(IVideo& vid) : IVideo(vid), base(vid.frames()) {};
    size_type frameCount() override { return base.frameCount(); };
    std::vector<Frame>& frames() & override { return base.frames(); };
    std::vector<std::unique_ptr<IScene>>& getScenes() & override { return emptyScenes; }
};

template<typename Base>
InputVideoAdapter<Base> make_video_adapter(Base&& b, const std::string& name) {
    return InputVideoAdapter<Base>(std::forward<Base>(b), name);
}

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
    inline bool shouldBaggifyFrames(IVideo& video) override { return true; };
    inline bool shouldComputeScenes(IVideo& video) override { return true; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return true; };
};

class LazyStorageStrategy : public IVideoStorageStrategy {
public:
    inline bool shouldBaggifyFrames(IVideo& video) override { return false; };
    inline bool shouldComputeScenes(IVideo& video) override { return false; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return false; };
};

class FileDatabase : public IDatabase {
private:
    fs::path databaseRoot;
    std::vector<std::unique_ptr<IVideo>> loadVideos() const;

public:
    FileDatabase(std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) : 
    FileDatabase(fs::current_path() / "database", std::move(strat), args) {};
    FileDatabase(const std::string& databasePath,
        std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) 
        : IDatabase(std::move(strat), args), databaseRoot(databasePath) {};

    std::unique_ptr<IVideo> saveVideo(IVideo& video) override;
    std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const override;
    bool saveVocab(const IVocab& vocab, const std::string& key) override;
    std::unique_ptr<IVocab> loadVocab(const std::string& key) const override;
};

inline std::unique_ptr<IDatabase> database_factory(const std::string& dbPath, int KFrame, int KScene) {
    return std::make_unique<FileDatabase>(dbPath, 
        std::make_unique<AggressiveStorageStrategy>(), 
        RuntimeArguments{KScene, KFrame});
}

#endif

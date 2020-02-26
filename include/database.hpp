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

namespace fs = std::experimental::filesystem;

class SIFTVideo;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);
SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});
cv::Mat scaleToTarget(cv::Mat image, int targetWidth, int targetHeight);

class IVideo {
public:
    IVideo(const std::string& name) : name(name) {};
    const std::string name;
    using size_type = std::vector<Frame>::size_type;

    virtual size_type frameCount() = 0;
    virtual std::vector<Frame>& frames() & = 0;
    virtual const std::vector<IScene>& getScenes() & = 0;
    virtual ~IVideo() = default;
};

class IScene {
public:
    IScene(const std::string& key) : key(key) {};
    virtual ~IScene() = default;
    const std::string key;

    template<typename Extractor> virtual std::invoke_result_t<Extractor, decltype<getFrames()>> getDescriptor(Extractor&& ext) {
        return ext(getFrames());
    }

    virtual vector<Frame>& getFrames() & = 0;
};

class SIFTVideo : public IVideo {
private:
    std::vector<Frame> SIFTFrames;
public:
    SIFTVideo(const std::string& name, const std::vector<Frame>& frames) : IVideo(name), SIFTFrames(frames) {};
    SIFTVideo(const std::string& name, std::vector<Frame>&& frames) : IVideo(name), SIFTFrames(frames) {};
    SIFTVideo(SIFTVideo&& vid) : IVideo(vid.name), SIFTFrames(vid.SIFTFrames) {};
    std::vector<Frame>& frames() & override { return SIFTFrames; };
    size_type frameCount() override { return SIFTFrames.size(); };
};

template<typename T>
class ICursor {
public:
    ICursor& advance() & = 0;
    operator bool()() = 0;
    const T& getValue() & const = 0;
};

class IDatabase {
public:
    virtual std::unique_ptr<IVideo> saveVideo(const IVideo& video) = 0;
    virtual ICursor<IVideo> loadVideo(const std::string& key = "") const = 0;
    template<typename Vocab> virtual saveVocab(Vocab&& vocab, const std::string& key) = 0;
    template<typename Vocab> virtual Vocab loadVocab(const std::string& key) = 0;
    virtual ~IDatabase() = default;
};

class IVideoLoadStrategy {
public:
    virtual std::unique_ptr<IVideo> operator()(const std::string& findKey) const = 0;
    virtual ~IVideoLoadStrategy() = default;
};

class IVideoStorageStrategy {
public:
    virtual IVideo& operator()(IVideo& video, IDatabase& database) const = 0;
    virtual ~IVideoStorageStrategy() = default;
};

class SubdirSearchStrategy : IVideoLoadStrategy {
public:
    std::unique_ptr<IVideo> operator()(const std::string& findKey) const;
};

class EagerStorageStrategy : IVideoStorageStrategy {
public:
    IVideo& operator()(IVideo& video, IDatabase& database) const ;
};

class FileDatabase : public IDatabase {
private:
    fs::path databaseRoot;
public:
    FileDatabase() : FileDatabase(fs::current_path() / "database") {};
    FileDatabase(const std::string& databasePath);
    std::unique_ptr<IVideo> addVideo(const std::string& filepath, std::function<void(cv::Mat, Frame)> callback = nullptr) override;
    std::unique_ptr<IVideo> loadVideo(const std::string& filepath) const override;
    std::vector<std::string> listVideos() const override;
};

#endif

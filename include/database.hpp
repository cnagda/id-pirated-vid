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

namespace fs = std::experimental::filesystem;

class SIFTVideo;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);
SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});
cv::Mat scaleToTarget(cv::Mat image, int targetWidth, int targetHeight);

typedef std::string hash_default;

class IVocab {
public:
    virtual std::string getHash() const = 0;
    virtual cv::Mat descriptors() const = 0;
    virtual ~IVocab() = default;
};

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
    std::string getHash() const override {
        return hash;
    }
    cv::Mat descriptors() const override {
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

template<typename T>
const std::string Vocab<T>::vocab_name = T::vocab_name;

class IScene {
public:
    static const std::string vocab_name;

    IScene(const std::string& key) : key(key) {};
    virtual ~IScene() = default;
    const std::string key;

    virtual cv::Mat descriptor() = 0;
    virtual std::vector<Frame>& getFrames() & = 0;
    virtual cv::Mat descriptor() const = 0;
    virtual const std::vector<Frame>& getFrames() const & = 0;
};

class IVideo {
public:
    IVideo(const std::string& name) : name(name) {};
    const std::string name;
    using size_type = std::vector<Frame>::size_type;

    virtual size_type frameCount() = 0;
    virtual std::vector<Frame>& frames() & = 0;
    virtual const std::vector<Frame>& frames() const & = 0;
    virtual std::vector<std::unique_ptr<IScene>>& getScenes() & = 0;
    virtual const std::vector<std::unique_ptr<IScene>>& getScenes() const & = 0;
    virtual ~IVideo() = default;
};

class SIFTVideo {
private:
    std::vector<Frame> SIFTFrames;
    using size_type = std::vector<Frame>::size_type;
public:
    SIFTVideo(const std::string& name, const std::vector<Frame>& frames) : name(name), SIFTFrames(frames) {};
    SIFTVideo(const std::string& name, std::vector<Frame>&& frames) : name(name), SIFTFrames(frames) {};
    SIFTVideo(SIFTVideo&& vid) : name(vid.name), SIFTFrames(vid.SIFTFrames) {};
    SIFTVideo(const SIFTVideo& vid) : name(vid.name), SIFTFrames(vid.SIFTFrames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    const std::vector<Frame>& frames() const & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };

    const std::string name;
};

template<typename Base>
class DatabaseVideo : public IVideo {
private:
    Base base;
    std::vector<std::unique_ptr<IScene>> scenes;
public:
    DatabaseVideo(Base&& b) : IVideo(b.name), base(b) {};
    DatabaseVideo(const Base& b) : IVideo(b.name), base(b) {};
    size_type frameCount() override { return base.frameCount(); };
    std::vector<Frame>& frames() & override { return base.frames(); };
    const std::vector<Frame>& frames() const & override { return base.frames(); };
    const std::vector<std::unique_ptr<IScene>>& getScenes() const & override { return scenes; };
    std::vector<std::unique_ptr<IScene>>& getScenes() & override { return scenes; }
};

template<typename T>
class ICursor {
public:
    virtual ICursor& advance() & = 0;
    virtual operator bool() = 0;
    virtual const T& getValue() const & = 0;
};

class IDatabase {
public:
    virtual std::unique_ptr<IVideo> saveVideo(const IVideo& video) = 0;
    virtual std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const = 0;
    virtual bool saveVocab(const IVocab& vocab, const std::string& key) = 0;
    virtual std::unique_ptr<IVocab> loadVocab(const std::string& key) const = 0;
    virtual ~IDatabase() = default;

    template<typename V>
    bool saveVocab(V&& vocab) { return saveVocab(std::forward<V>(vocab), V::vocab_name); }
    template<typename V>
    std::optional<V> loadVocab() const { 
        auto v = loadVocab(V::vocab_name);
        if(v) {
            return V(*v);
        }
        return std::nullopt;
    }
};

/* Example strategies
class IVideoLoadStrategy {
public:
    virtual std::unique_ptr<IVideo> operator()(const std::string& findKey) const = 0;
    virtual ~IVideoLoadStrategy() = default;
};

class IVideoStorageStrategy {
public:
    virtual IVideo& operator()(IVideo& video, IDatabase& database) const = 0;
    virtual ~IVideoStorageStrategy() = default;
}; */

class SubdirSearchStrategy {
    fs::path directory;
public:
    SubdirSearchStrategy() : SubdirSearchStrategy(fs::current_path()) {};
    SubdirSearchStrategy(const std::string& path) : directory(path) {};

    template<typename FileReader, typename ...Args>
    auto operator()(const std::string& findKey, FileReader &&reader, Args&&... args) const {
        return std::invoke(reader, directory / findKey, args...);
    }
};

// TODO implement this better
/* class LazyStorageStrategy {
public:
    IVideo& saveVideo(IVideo& video, IDatabase& database) const;
};

class LazyLoadStrategy {
public:
    IVideo& loadVideo(IVideo& video, IDatabase& database) const;
}; */

class EagerStorageStrategy {
public:
    IVideo& saveVideo(IVideo& video, IDatabase& database) const;
};

class EagerLoadStrategy {
public:
    IVideo& loadVideo(IVideo& video, IDatabase& database) const;
};

class FileDatabase : public IDatabase {
private:
    fs::path databaseRoot;
    std::vector<std::unique_ptr<IVideo>> loadVideos() const;
public:
    FileDatabase() : FileDatabase(fs::current_path() / "database") {};
    FileDatabase(const std::string& databasePath) : databaseRoot(databasePath) {};
    std::unique_ptr<IVideo> saveVideo(const IVideo& video) override;
    std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const override;
    bool saveVocab(const IVocab& vocab, const std::string& key) override;
    std::unique_ptr<IVocab> loadVocab(const std::string& key) const override;
};

inline std::unique_ptr<IDatabase> database_factory(const std::string& dbPath) {
    return std::make_unique<FileDatabase>(dbPath);
}

#endif

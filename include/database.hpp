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

typedef std::string hash_default;

template<typename Hash = hash_default, typename Matrix = cv::Mat>
class IVocab {
public:
    virtual Hash getHash() const;
    virtual Matrix descriptors() const;
};

template<typename T, typename Hash = hash_default, typename Matrix = cv::Mat>
class LazyVocab : IVocab<Hash, Matrix>{
private:
    fs::path directory;
public:
    LazyVocab(fs::path directory) : directory(directory) {};
    Hash getHash() const override {
    }
    Matrix descriptors() const override {
        Matrix myvocab;
        cv::FileStorage fs(directory / T::vocab_name, FileStorage::READ);
        fs["Vocabulary"] >> myvocab;
        return myvocab;
    }
    static const std::string vocab_name = T::vocab_name;
};

template<typename T, typename Hash = hash_default, typename Matrix = cv::Mat>
class Vocab : IVocab<Hash, Matrix>{
private:
    const Matrix descriptors;
    Hash hash;
public:
    FileVocab(Matrix&& descriptors) : descriptors(descriptors) {};
    Hash getHash() const override {
        return hash;
    }
    Matrix descriptors() const override {
        return descriptors;
    }
    static const std::string vocab_name = T::vocab_name;
};

class IScene {
public:
    static const std::string vocab_name = "SceneVocab.mat";

    IScene(const std::string& key) : key(key) {};
    virtual ~IScene() = default;
    const std::string key;

    template<typename Extractor> auto getDescriptor(Extractor&& ext) {
        return ext(getFrames());
    }

    virtual std::vector<Frame>& getFrames() & = 0;
};

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

class SIFTVideo {
private:
    std::vector<Frame> SIFTFrames;
    using size_type = std::vector<Frame>::size_type;
    const std::string name;
public:
    SIFTVideo(const std::string& name, const std::vector<Frame>& frames) : name(name), SIFTFrames(frames) {};
    SIFTVideo(const std::string& name, std::vector<Frame>&& frames) : name(name), SIFTFrames(frames) {};
    SIFTVideo(SIFTVideo&& vid) : name(vid.name), SIFTFrames(vid.SIFTFrames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

template<typename Base>
class DatabaseVideo : public IVideo {
private:
    Base base;
    std::vector<IScene> scenes;
public:
    DatabaseVideo(Base&& base) : DatabaseVideo(base.name), base(base) {};
    size_type frameCount() override {
        return base.frameCount();
    };
    std::vector<Frame>& frames() & override {
        return base.frames();
    };
    const std::vector<IScene>& getScenes() & override {
        return scenes;
    };
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
    template<typename V>
    bool saveVocab(const V& vocab) { return saveVocab(vocab, V::vocab_name); }
    template<typename V>
    V loadVocab() { return V(loadVocab(V::vocab_name)); }
    virtual bool saveVocab(const IVocab& vocab, const std::string& key) = 0;
    virtual IVocab loadVocab(const std::string& key) = 0;
    virtual ~IDatabase() = default;
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

    template<typename FileVideoReader, typename ...Args>
    auto operator()(const std::string& findKey, FileReader &&reader, Args&&... args) const {
        return reader(directory / findKey, args...);
    }
};

class LazyStorageStrategy {
public:
    IVideo& operator()(IVideo& video, IDatabase& database) const;
}

class EagerStorageStrategy {
public:
    IVideo& operator()(IVideo& video, IDatabase& database) const;
};

class FileDatabase : public IDatabase {
private:
    fs::path databaseRoot;
public:
    FileDatabase() : FileDatabase(fs::current_path() / "database") {};
    FileDatabase(const std::string& databasePath);
    virtual std::unique_ptr<IVideo> saveVideo(const IVideo& video) override;
    virtual std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const override;
    IVocab saveVocab(const IVocab& vocab, const std::string& key) override;
    cv::Mat loadVocab(const std::string& key) override;
};

inline std::unique_ptr<IDatabase> database_factory(const std::string& dbPath) {
    return std::make_unique<FileDatabase>(dbPath);
}

#endif

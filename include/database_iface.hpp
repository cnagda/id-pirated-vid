#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "frame.hpp"
#include <exception>

class IVocab {
public:
    virtual std::string getHash() const = 0;
    virtual cv::Mat descriptors() const = 0;
    virtual ~IVocab() = default;
};

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

struct RuntimeArguments {
    int KScenes;
    int KFrame;
};

class IDatabase {
protected:
    std::unique_ptr<IVideoStorageStrategy> strategy;
    RuntimeArguments args;
    IDatabase(std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) : strategy(std::move(strat)), args(args) {};
public:
    virtual std::unique_ptr<IVideo> saveVideo(IVideo& video) = 0;
    virtual std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const = 0;
    virtual bool saveVocab(const IVocab& vocab, const std::string& key) = 0;
    virtual std::unique_ptr<IVocab> loadVocab(const std::string& key) const = 0;
    virtual ~IDatabase() = default;
};

template<typename V, typename Db>
bool saveVocabulary(V&& vocab, Db&& db) { 
    return db.saveVocab(std::forward<V>(vocab), std::remove_reference_t<V>::vocab_name); 
}

template<typename V, typename Db>
std::optional<V> loadVocabulary(Db&& db) { 
    auto v = db.loadVocab(V::vocab_name);
    if(v) {
        return V(*v);
    }
    return std::nullopt;
}

#endif
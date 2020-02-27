#ifndef DATABASE_ERASURE_HPP
#define DATABASE_ERASURE_HPP
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

class IVocab {
public:
    virtual std::string getHash() const = 0;
    virtual cv::Mat descriptors() const = 0;
    virtual ~IVocab() = default;
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

class IDatabase {
public:
    virtual std::unique_ptr<IVideo> saveVideo(IVideo& video) = 0;
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

#endif
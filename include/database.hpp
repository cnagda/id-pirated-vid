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

struct RuntimeArguments {
    int KScenes;
    int KFrame;
};

class ContainerVocab {
private:
    cv::Mat desc;
    std::string hash;
public:
    ContainerVocab() = default;
    ContainerVocab(const cv::Mat& descriptors) : desc(descriptors), hash() {};
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
    Vocab(const ContainerVocab& v) : ContainerVocab(v) {};
    static const std::string vocab_name;
    typedef T vocab_type;
};

class SIFTVideo {
private:
    std::vector<Frame> SIFTFrames;
    using size_type = std::vector<Frame>::size_type;
public:
    SIFTVideo(const std::vector<Frame>& frames) : SIFTFrames(frames) {};
    SIFTVideo(std::vector<Frame>&& frames) : SIFTFrames(frames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

template<typename Base>
class InputVideoAdapter : public IVideo {
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

class FileDatabase {
private:
    fs::path databaseRoot;
    std::vector<std::unique_ptr<IVideo>> loadVideos() const;
    std::unique_ptr<IVideoStorageStrategy> strategy;
    RuntimeArguments args;
public:
    FileDatabase(std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) : 
    FileDatabase(fs::current_path() / "database", std::move(strat), args) {};

    FileDatabase(const std::string& databasePath,
        std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) 
        : strategy(std::move(strat)), args(args), databaseRoot(databasePath) {};

    std::unique_ptr<IVideo> saveVideo(IVideo& video);
    std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const;
    bool saveVocab(const ContainerVocab& vocab, const std::string& key);
    std::optional<ContainerVocab> loadVocab(const std::string& key) const;
};

inline std::unique_ptr<FileDatabase> database_factory(const std::string& dbPath, int KFrame, int KScene) {
    return std::make_unique<FileDatabase>(dbPath, 
        std::make_unique<AggressiveStorageStrategy>(), 
        RuntimeArguments{KScene, KFrame});
}

template<typename V, typename Db>
bool saveVocabulary(V&& vocab, Db&& db) { 
    return db.saveVocab(std::forward<V>(vocab), std::remove_reference_t<V>::vocab_name); 
}

template<typename V, typename Db>
std::optional<V> loadVocabulary(Db&& db) { 
    auto v = db.loadVocab(V::vocab_name);
    if(v) {
        return V(v.value());
    }
    return std::nullopt;
}

template<typename V, typename Db>
V loadOrComputeVocab(Db&& db, int K) {
    auto vocab = loadVocabulary<V>(std::forward<Db>(db));
    if(!vocab) {
        if(K == -1) {
            throw std::runtime_error("need to compute " + V::vocab_name + " vocabulary but not K provided");
        }

        V v;
        if constexpr(std::is_same_v<typename V::vocab_type, Frame>) {
            v = constructFrameVocabulary(db, K);
        } else if constexpr(std::is_base_of_v<typename V::vocab_type, IScene>) {
            v = constructSceneVocabulary(db, K);
        }
        saveVocabulary(std::forward<V>(v), std::forward<Db>(db));
        return v;
    }
    return vocab.value();
}

#endif

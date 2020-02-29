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
#include "vocabulary.hpp"
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

namespace fs = std::experimental::filesystem;

class SIFTVideo;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);
SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});
cv::Mat scaleToTarget(cv::Mat image, int targetWidth, int targetHeight);


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

struct RuntimeArguments {
    int KScenes;
    int KFrame;
};

struct Configuration {
    int KScenes;
    int KFrames;
    int threshold;
    SaveStrategyType strategy;

    Configuration(const RuntimeArguments& args, SaveStrategyType type)
        : KScenes(args.KScenes), KFrames(args.KFrame), threshold(0.2), strategy(type) {};
};

struct SIFTVideo {
    using size_type = std::vector<Frame>::size_type;

    std::vector<Frame> SIFTFrames;
    SIFTVideo(const std::vector<Frame>& frames) : SIFTFrames(frames) {};
    SIFTVideo(std::vector<Frame>&& frames) : SIFTFrames(frames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

template<typename Base>
class InputVideoAdapter : public IVideo {
private:
    Base base;
    std::vector<std::shared_ptr<IScene>> emptyScenes;
public:
    using size_type = typename Base::size_type;

    explicit InputVideoAdapter(Base&& b, const std::string& name) : IVideo(name), base(std::forward<Base>(b)) {};
    explicit InputVideoAdapter(const Base& b, const std::string& name) : IVideo(name), base(std::forward<Base>(b)) {};
    InputVideoAdapter(IVideo&& vid) : IVideo(vid), base(vid.frames()) {};
    InputVideoAdapter(IVideo& vid) : IVideo(vid), base(vid.frames()) {};
    size_type frameCount() override { return base.frameCount(); };
    std::vector<Frame>& frames() & override { return base.frames(); };
    std::vector<std::shared_ptr<IScene>>& getScenes() & override { return emptyScenes; }
};

struct SerializableScene {
    cv::Mat frameBag;
    SIFTVideo::size_type startIdx, endIdx;

    template<typename Video>
    auto getFrameRange(Video& video) const {
        auto frames = video.frames();
        return std::make_pair(frames.begin() + startIdx, frames.begin() + endIdx);
    };

    template<typename It>
    auto getFrameRange(It begin, std::random_access_iterator_tag) const {
        return std::make_pair(begin + startIdx, begin + endIdx);
    };
};

class FileLoader {
private:
    fs::path rootDir;
public:
    explicit FileLoader(fs::path dir) : rootDir(dir) {};

    std::optional<Frame> readFrame(const std::string& videoName, SIFTVideo::size_type index) const;
    std::optional<SerializableScene> readScene(const std::string& videoName, SIFTVideo::size_type index) const;
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
    inline SaveStrategyType getType() const { return Eager; };
    inline bool shouldBaggifyFrames(IVideo& video) override { return true; };
    inline bool shouldComputeScenes(IVideo& video) override { return true; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return true; };
};

class LazyStorageStrategy : public IVideoStorageStrategy {
public:
    inline SaveStrategyType getType() const { return Lazy; };
    inline bool shouldBaggifyFrames(IVideo& video) override { return false; };
    inline bool shouldComputeScenes(IVideo& video) override { return false; };
    inline bool shouldBaggifyScenes(IVideo& video) override { return false; };
};

class FileDatabase {
private:
    fs::path databaseRoot;
    std::vector<std::unique_ptr<IVideo>> loadVideos() const;
    std::unique_ptr<IVideoStorageStrategy> strategy;
    Configuration config;
    FileLoader loader;
public:
    explicit FileDatabase(std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) : 
    FileDatabase(fs::current_path() / "database", std::move(strat), args) {};

    explicit FileDatabase(const std::string& databasePath,
        std::unique_ptr<IVideoStorageStrategy>&& strat, RuntimeArguments args) 
        : strategy(std::move(strat)), config(args, strategy->getType()), databaseRoot(databasePath), 
        loader(databasePath) {};

    std::unique_ptr<IVideo> saveVideo(IVideo& video);
    std::vector<std::unique_ptr<IVideo>> loadVideo(const std::string& key = "") const;

    const FileLoader& getFileLoader() const & { return loader; };
    const Configuration& getConfig() const & { return config; };

    bool saveVocab(const ContainerVocab& vocab, const std::string& key);
    std::optional<ContainerVocab> loadVocab(const std::string& key) const;
};


class DatabaseScene : public IScene {
    static_assert(std::is_convertible_v<std::shared_ptr<DatabaseScene>, std::shared_ptr<IScene>>, "not convertible");
    std::vector<Frame> frames;
    cv::Mat descriptorCache;
    const IVideo& video;
    const FileDatabase& database;

public:
    DatabaseScene() = delete;
    template<typename Range>
    explicit DatabaseScene(IVideo& video, const FileDatabase& database, Range frames) :
    video(video), database(database), frames(), descriptorCache() {
        boost::push_back(this->frames, frames);
    };
    
    explicit DatabaseScene(IVideo& video, const FileDatabase& database, const SerializableScene& scene) :
    video(video), database(database), frames(frames) {
        auto frames = video.frames();
        auto vocab = loadVocabulary<Vocab<DatabaseScene>>(database);
        
        if(!vocab) {
            throw std::runtime_error("Scene couldn't get a frame vocabulary");
        }

        auto range = scene.getFrameRange(video) | 
            boost::adaptors::transformed([](auto f) { 
            return f.descriptors; });
            
        descriptorCache = baggify(range.begin(), range.end(), 
            vocab->descriptors());
    };

    const cv::Mat& descriptor() override {
        if(descriptorCache.empty()) {
            auto frames = getFrames();
            auto vocab = loadVocabulary<Vocab<DatabaseScene>>(database);
            if(!vocab) {
                throw std::runtime_error("Scene couldn't get a frame vocabulary");
            }
            auto access = [](auto frame){ return frame.descriptors; };
            descriptorCache = baggify(
                boost::make_transform_iterator(frames.begin(), access), 
                boost::make_transform_iterator(frames.end(), access), 
                vocab->descriptors());
        }

        return descriptorCache;
    }

    const std::vector<Frame>& getFrames() override {
        return frames;
    }
};

class DatabaseVideo : public IVideo {
    const FileDatabase& db;
    std::vector<std::shared_ptr<IScene>> sceneCache;
    std::vector<Frame> frameCache;
public:
    DatabaseVideo() = delete;
    explicit DatabaseVideo(const FileDatabase& database, const std::string& key, const std::vector<Frame>& frames) : IVideo(key), 
    db(database), frameCache(frames), sceneCache() {};
    explicit DatabaseVideo(const FileDatabase& database, const std::string& key, const std::vector<Frame>& frames, const std::vector<SerializableScene>& scenes) : IVideo(key), 
    db(database), frameCache(frames) {
        boost::push_back(sceneCache, scenes | boost::adaptors::transformed(
            [this](auto scene){ return std::make_shared<DatabaseScene>(*this, db, scene); }
        ));
    };


    size_type frameCount() override { return frames().size(); };
    std::vector<Frame>& frames() & override { return frameCache; };

    std::vector<std::shared_ptr<IScene>>& getScenes() & override;
};

inline std::unique_ptr<FileDatabase> database_factory(const std::string& dbPath, int KFrame, int KScene) {
    return std::make_unique<FileDatabase>(dbPath, 
        std::make_unique<AggressiveStorageStrategy>(), 
        RuntimeArguments{KScene, KFrame});
}

#endif

#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"
#include <vector>
#include <memory>
#include <experimental/filesystem>
#include <optional>
#include "database_iface.hpp"
#include "vocab_type.hpp"
#include "storage.hpp"
#include <variant>

namespace fs = std::experimental::filesystem;

std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);

struct RuntimeArguments {
    int KScenes;
    int KFrame;
    double threshold;
};

struct Configuration {
    int KScenes;
    int KFrames;
    double threshold;
    StrategyType storageStrategy, loadStrategy;

    Configuration() : KScenes(-1), KFrames(-1), threshold(-1) {};
    Configuration(const RuntimeArguments& args, StrategyType storage, StrategyType load)
        : KScenes(args.KScenes), KFrames(args.KFrame), threshold(args.threshold), 
        storageStrategy(storage), loadStrategy(load) {};
};

template<typename Base>
class InputVideoAdapter : public IVideo {
private:
    typename std::decay_t<Base> base;
    std::vector<SerializableScene> emptyScenes;
public:
    using size_type = typename std::decay_t<Base>::size_type;

    InputVideoAdapter(Base&& b, const std::string& name) : IVideo(name), base(std::forward<Base>(b)) {};
    InputVideoAdapter(IVideo&& vid) : IVideo(vid), base(vid.frames()) {};
    InputVideoAdapter(IVideo& vid) : IVideo(vid), base(vid.frames()) {};
    size_type frameCount() override { return base.frameCount(); };
    std::vector<Frame>& frames() & override { return base.frames(); };
    std::vector<SerializableScene>& getScenes() & override { return emptyScenes; }
};

class DatabaseVideo;

class FileDatabase {
private:
    fs::path databaseRoot;
    std::vector<std::unique_ptr<IVideo>> loadVideos() const;
    std::unique_ptr<IVideoStorageStrategy> strategy;
    Configuration config;
    FileLoader loader;

    typedef StrategyType LoadStrategy;
    LoadStrategy loadStrategy;
public:
    FileDatabase(std::unique_ptr<IVideoStorageStrategy>&& strat, LoadStrategy l, RuntimeArguments args) :
    FileDatabase(fs::current_path() / "database", std::move(strat), l, args) {};

    FileDatabase(const std::string& databasePath,
        std::unique_ptr<IVideoStorageStrategy>&& strat, LoadStrategy l, RuntimeArguments args);

    std::optional<DatabaseVideo> saveVideo(IVideo& video);
    std::optional<DatabaseVideo> loadVideo(const std::string& key = "") const;
    std::vector<std::string> listVideos() const;

    const FileLoader& getFileLoader() const & { return loader; };
    const Configuration& getConfig() const & { return config; };

    bool saveVocab(const ContainerVocab& vocab, const std::string& key);
    std::optional<ContainerVocab> loadVocab(const std::string& key) const;
};

class DatabaseVideo : public IVideo {
    const FileDatabase& db;
    std::vector<SerializableScene> sceneCache;
    std::vector<Frame> frameCache;
public:
    DatabaseVideo() = delete;
    DatabaseVideo(const FileDatabase& database, const std::string& key) : 
        DatabaseVideo(database, key, {}, {}) {};
    DatabaseVideo(const FileDatabase& database, const std::string& key, const std::vector<Frame>& frames) :
        DatabaseVideo(database, key, frames, {}) {};
    DatabaseVideo(const FileDatabase& database, const std::string& key, const std::vector<SerializableScene> scenes) :
        DatabaseVideo(database, key, {}, scenes) {};
    DatabaseVideo(const FileDatabase& database, const std::string& key, const std::vector<Frame>& frames, const std::vector<SerializableScene>& scenes) : IVideo(key),
        db(database), frameCache(frames), sceneCache(scenes) {};


    inline size_type frameCount() override { return frameCache.size(); };
    std::vector<Frame>& frames() & override;

    std::vector<SerializableScene>& getScenes() & override;
};

inline std::unique_ptr<FileDatabase> database_factory(const std::string& dbPath, int KFrame, int KScene, double threshold) {
    return std::make_unique<FileDatabase>(dbPath,
        std::make_unique<AggressiveStorageStrategy>(),
        LazyLoadStrategy{},
        RuntimeArguments{KScene, KFrame, threshold});
}

inline std::unique_ptr<FileDatabase> query_database_factory(const std::string& dbPath, int KFrame, int KScene, double threshold) {
    return std::make_unique<FileDatabase>(dbPath,
        std::make_unique<LazyStorageStrategy>(),
        LazyLoadStrategy{},
        RuntimeArguments{KScene, KFrame, threshold});
}


DatabaseVideo make_scene_adapter(FileDatabase& db, IVideo& video, const std::string& key);

template<typename Video>
inline DatabaseVideo make_query_adapter(FileDatabase& db, Video&& video, const std::string& key) {
    auto frames = video.frames();
    return {db, key, frames};
}


template<typename Base>
InputVideoAdapter<Base> make_video_adapter(Base&& b, const std::string& name) {
    return {std::forward<Base>(b), name};
}

#endif

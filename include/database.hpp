#ifndef DATABASE_H
#define DATABASE_H
#include <vector>
#include <memory>
#include <experimental/filesystem>
#include <optional>
#include "database_iface.hpp"
#include "vocab_type.hpp"
#include "storage.hpp"

namespace fs = std::experimental::filesystem;

std::string getAlphas(const std::string &input);
void createFolder(const std::string &folder_name);
std::string to_string(const fs::path &);

struct RuntimeArguments
{
    int KScenes;
    int KFrame;
    double threshold;
};

struct Configuration
{
    int KScenes;
    int KFrames;
    double threshold;
    StrategyType storageStrategy, loadStrategy;

    Configuration() : KScenes(-1), KFrames(-1), threshold(-1){};
    Configuration(const RuntimeArguments &args, StrategyType storage, StrategyType load)
        : KScenes(args.KScenes), KFrames(args.KFrame), threshold(args.threshold),
          storageStrategy(storage), loadStrategy(load){};
};

class DatabaseVideo;

struct DatabaseMetadata {
    size_t frameHash, sceneHash;

    bool operator==(const DatabaseMetadata& other) const;
};

struct VideoMetadata : public DatabaseMetadata {};

class FileDatabase
{
private:
    Configuration config;
    FileLoader loader;

public:
    const fs::path databaseRoot;
    std::unique_ptr<IVideoStorageStrategy> strategy;
    StrategyType loadStrategy;

    FileDatabase(std::unique_ptr<IVideoStorageStrategy> &&strat, StrategyType l, RuntimeArguments args) : FileDatabase(to_string(fs::current_path() / "database"), std::move(strat), l, args){};

    FileDatabase(const std::string &databasePath,
                 std::unique_ptr<IVideoStorageStrategy> &&strat, StrategyType l, RuntimeArguments args);

    std::optional<DatabaseVideo> saveVideo(const DatabaseVideo& video);
    std::optional<DatabaseVideo> saveVideo(const SIFTVideo& video, const std::string& key);

    std::optional<DatabaseVideo> loadVideo(const std::string &key) const;
    std::vector<std::string> listVideos() const;

    const FileLoader &getFileLoader() const & { return loader; };
    const Configuration &getConfig() const & { return config; };

    bool saveVocab(const ContainerVocab &vocab, const std::string &key);
    std::optional<ContainerVocab> loadVocab(const std::string &key) const;

    DatabaseMetadata loadMetadata() const;
};

class DatabaseVideo : public IVideo
{
    const FileDatabase &db;
    fs::path videoRoot;
    StrategyType loadStrategy;

public:
    DatabaseVideo() = delete;
    DatabaseVideo(const FileDatabase &database, const fs::path& videoRoot, const std::string &key) : IVideo(key), db(database), videoRoot(videoRoot) {};

    auto frames() const {
       return make_frame_source(db.getFileLoader(), name);
    }

    auto getScenes() const {
        return make_scene_source(db.getFileLoader(), name);
    }

    VideoMetadata getMetadata() const;
};

inline std::unique_ptr<FileDatabase> database_factory(const std::string &dbPath, int KFrame, int KScene, double threshold)
{
    return std::make_unique<FileDatabase>(dbPath,
                                          std::make_unique<AggressiveStorageStrategy>(),
                                          LazyLoadStrategy{},
                                          RuntimeArguments{KScene, KFrame, threshold});
}

inline std::unique_ptr<FileDatabase> query_database_factory(const std::string &dbPath, int KFrame, int KScene, double threshold)
{
    return std::make_unique<FileDatabase>(dbPath,
                                          std::make_unique<LazyStorageStrategy>(),
                                          LazyLoadStrategy{},
                                          RuntimeArguments{KScene, KFrame, threshold});
}

#endif

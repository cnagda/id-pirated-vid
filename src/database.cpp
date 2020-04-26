#include "database.hpp"
#include <cctype>
#include <experimental/filesystem>
#include <memory>
#include "vocabulary.hpp"
#include "matcher.hpp"
#include "video.hpp"
#include "scene_detector.hpp"
#include "imgproc.hpp"
#include "filter.hpp"
#include <tbb/pipeline.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <atomic>

#define VIDEO_METADATA_FILENAME "metadata.bin"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

size_t getHash(ifstream &input)
{
    if (!input.is_open())
    {
        return 0;
    }

    std::string read;
    auto pos = input.tellg();
    input.seekg(0, ios::end);
    read.reserve(input.tellg());

    input.seekg(pos);
    read.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());

    return std::hash<std::string>{}(read);
}

class CaptureSource : public ICursor<cv::UMat>
{
    VideoCapture cap;

public:
    CaptureSource(const std::string &filename) : cap(filename, cv::CAP_ANY) {}

    std::optional<cv::UMat> read() override
    {
        UMat image;

        if (cap.read(image))
        {
            return image;
        }

        return std::nullopt;
    }
};

class FrameSource : public ICursor<Frame>
{
    CaptureSource source;
    std::function<void(UMat, Frame)> callback;
    ScaleImage scale;
    ExtractColorHistogram color;
    ExtractSIFT sift;

public:
    FrameSource(const std::string &filename, std::function<void(UMat, Frame)> callback,
                std::pair<int, int> cropsize) : source(filename), callback(callback), scale(cropsize){};

    std::optional<Frame> read() override
    {
        if (auto image = source.read())
        {
            auto ordered = ordered_umat{0, *image};
            auto scaled = scale(ordered);
            auto colorHistogram = color(scaled);
            auto descriptors = sift(scaled);
            Frame frame{descriptors.data, cv::Mat(), colorHistogram.data};
            if (callback)
                callback(*image, frame);

            return frame;
        }
        return std::nullopt;
    }
};

std::unique_ptr<ICursor<cv::UMat>> SIFTVideo::images() const
{
    return std::make_unique<CaptureSource>(filename);
}

std::unique_ptr<ICursor<Frame>> SIFTVideo::frames() const
{
    return std::make_unique<FrameSource>(filename, callback, cropsize);
}

SIFTVideo::SIFTVideo(const std::string &filename, std::function<void(cv::UMat, Frame)> callback, std::pair<int, int> cropsize)
    : IVideo(fs::path(filename).filename()), filename(filename), callback(callback), cropsize(cropsize) {}

SIFTVideo getSIFTVideo(const std::string &filepath, std::function<void(UMat, Frame)> callback, std::pair<int, int> cropsize)
{
    return {filepath, callback, cropsize};
}

const std::string SerializableScene::vocab_name = "SceneVocab.mat";
const std::string Frame::vocab_name = "FrameVocab.mat";

template <typename T>
const std::string Vocab<T>::vocab_name = T::vocab_name;

template <typename Read>
class frame_bag_adapter : public ICursor<Frame>
{
    std::decay_t<Read> reader;
    Vocab<Frame> vocab;

public:
    frame_bag_adapter(Read &&r, const Vocab<Frame> &v) : reader(std::move(r)), vocab(v) {}
    frame_bag_adapter(const Read &r, const Vocab<Frame> &v) : reader(r), vocab(v) {}

    constexpr std::optional<Frame> read() override
    {
        if (auto val = reader.read())
        {
            loadFrameDescriptor(*val, vocab.descriptors());
            return val;
        }
        return std::nullopt;
    }
};

template <typename SceneRead, typename FrameRead>
class scene_bag_adapter : public ICursor<SerializableScene>
{
    std::decay_t<SceneRead> scene_reader;
    std::decay_t<FrameRead> frame_reader;

    Vocab<SerializableScene> vocab;

public:
    scene_bag_adapter(SceneRead &&s, FrameRead &&f, const Vocab<SerializableScene> &v) : scene_reader(std::move(s)), frame_reader(std::move(f)), vocab(v) {}
    scene_bag_adapter(const SceneRead &s, const FrameRead &f, const Vocab<SerializableScene> &v) : scene_reader(s), frame_reader(f), vocab(v) {}
    scene_bag_adapter(const SceneRead &s, FrameRead &&f, const Vocab<SerializableScene> &v) : scene_reader(s), frame_reader(std::move(f)), vocab(v) {}
    scene_bag_adapter(SceneRead &&s, const FrameRead &f, const Vocab<SerializableScene> &v) : scene_reader(std::move(s)), frame_reader(f), vocab(v) {}

    constexpr std::optional<SerializableScene> read() override
    {
        if (auto val = scene_reader.read())
        {
            return val;
        }
        return std::nullopt;
    }
};

template <typename Read>
class scene_detect_cursor : ICursor<SerializableScene>
{
    std::vector<std::pair<unsigned int, unsigned int>> scenes;
    decltype(scenes.begin()) iterator;

public:
    scene_detect_cursor() : iterator(scenes.begin()) {}
    scene_detect_cursor(Read &reader, unsigned int min_scenes)
    {
        scenes = hierarchicalScenes(get_distances(reader, ColorComparator{}), min_scenes);
        iterator = scenes.begin();
    }
    scene_detect_cursor(Read &&reader, unsigned int min_scenes)
    {
        scenes = hierarchicalScenes(get_distances(std::move(reader), ColorComparator{}), min_scenes);
        iterator = scenes.begin();
    }

    constexpr std::optional<SerializableScene> read() override
    {
        if (iterator == scenes.end())
        {
            return std::nullopt;
        }
        return SerializableScene{*iterator++};
    }
};

std::unique_ptr<ICursor<Frame>> DatabaseVideo::frames() const
{
    auto frame_source = make_frame_source(db.getFileLoader(), name);
    auto vocab = loadVocabulary<Vocab<Frame>>(db);

    if (loadStrategy == Eager &&
        db.loadMetadata().frameHash != loadMetadata().frameHash &&
        vocab)
    {
        frame_bag_adapter source{frame_source, *vocab};
        return std::make_unique<decltype(source)>(std::move(source));
    }
    else
    {
        return std::make_unique<decltype(frame_source)>(std::move(frame_source));
    }
}

std::unique_ptr<ICursor<SerializableScene>> DatabaseVideo::getScenes() const
{
    auto scene_source = make_scene_source(db.getFileLoader(), name);
    auto vocab = loadVocabulary<Vocab<SerializableScene>>(db);
    if (loadStrategy == Eager &&
        db.loadMetadata() != loadMetadata() &&
        vocab)
    {
        scene_bag_adapter source{scene_source, frames(), *vocab};
        return std::make_unique<decltype(source)>(std::move(source));
    }
    else
    {
        return std::make_unique<decltype(scene_source)>(std::move(scene_source));
    }
}

FileDatabase::FileDatabase(const std::string &databasePath,
                           std::unique_ptr<IVideoStorageStrategy> &&strat, StrategyType l, RuntimeArguments args)
    : config(args, strat->getType(), l),
      loader(databasePath),
      strategy(std::move(strat)),
      loadStrategy(l),
      databaseRoot(databasePath)
{
    if (!fs::exists(databaseRoot))
    {
        fs::create_directories(databaseRoot);
    }
    Configuration configFromFile;
    std::ifstream reader(databaseRoot / "config.bin", std::ifstream::binary);
    if (reader.is_open())
    {
        reader.read((char *)&configFromFile, sizeof(configFromFile));
        if (config.threshold == -1)
        {
            config.threshold = configFromFile.threshold;
        }
        if (config.KScenes == -1)
        {
            config.KScenes = configFromFile.KScenes;
        }
        if (config.KFrames == -1)
        {
            config.KFrames = configFromFile.KFrames;
        }
    }

    std::ofstream writer(databaseRoot / "config.bin", std::ofstream::binary);
    writer.write((char *)&config, sizeof(config));
};

auto make_image_source(std::unique_ptr<ICursor<UMat>> source)
{
    size_t index = 0;
    return [source = std::move(source), index](tbb::flow_control &fc) mutable {
        if (auto image = source->read())
        {
            return ordered_umat{index++, *image};
        }
        fc.stop();
        return ordered_umat{};
    };
}

void writeMetadata(const VideoMetadata &data, const fs::path &videoDir)
{
    ofstream stream(videoDir / VIDEO_METADATA_FILENAME);
    stream.write((char *)&data, sizeof(data));
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const SIFTVideo &video, const ::string &key)
{
    VideoMetadata metadata{};
    fs::path video_dir{databaseRoot / key};
    loader.initVideoDir(key);
    loader.clearFrames(key);

    auto source = make_image_source(video.images());

    ScaleImage scale(video.cropsize);
    ExtractSIFT sift;
    ExtractColorHistogram color;
    SaveFrameSink saveFrame(key, getFileLoader());

    auto filter = tbb::make_filter<void, ordered_umat>(tbb::filter::serial_out_of_order, [&source](auto fc) {
                      return source(fc);
                  }) &
                  tbb::make_filter<ordered_umat, ordered_umat>(tbb::filter::parallel, scale) & tbb::make_filter<ordered_umat, std::pair<Frame, ordered_umat>>(tbb::filter::parallel, [&](auto mat) {
                      Frame f;
                      f.descriptors = sift(mat).data;
                      return make_pair(f, mat);
                  }) &
                  tbb::make_filter<std::pair<Frame, ordered_umat>, ordered_frame>(tbb::filter::parallel, [&](auto pair) {
                      pair.first.colorHistogram = color(pair.second).data;
                      return ordered_frame{pair.second.rank, pair.first};
                  });

    if (strategy->shouldBaggifyFrames())
    {
        if (auto vocab = loadVocabulary<Vocab<Frame>>(*this))
        {
            metadata.frameHash = loadMetadata().frameHash;
            ExtractFrame frame(*vocab);
            filter = filter & tbb::make_filter<ordered_frame, ordered_frame>(tbb::filter::parallel, [frame](auto f) {
                         f.data.frameDescriptor = frame(f.data.descriptors);
                         return f;
                     });
        }
    }

    std::atomic<size_t> frameCount = 0;

    tbb::parallel_pipeline(300,
                           filter &
                               tbb::make_filter<ordered_frame, void>(tbb::filter::parallel, [&](auto frame) {
                                   frameCount.fetch_add(1, std::memory_order_relaxed);
                                   saveFrame(frame);
                               }));

    metadata.frameCount = frameCount.load(std::memory_order_relaxed);
    writeMetadata(metadata, video_dir);
    return saveVideo(DatabaseVideo{*this, key});
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const DatabaseVideo &video)
{
    auto vocab = loadVocabulary<Vocab<Frame>>(*this);
    auto sceneVocab = loadVocabulary<Vocab<SerializableScene>>(*this);
    auto metadata = video.loadMetadata();

    if (strategy->shouldBaggifyFrames())
    {
    }
    if (strategy->shouldComputeScenes())
    {
        if (strategy->shouldBaggifyScenes())
        {
        }
    }

    writeMetadata(metadata, databaseRoot / video.name);
    return DatabaseVideo(*this, video.name);
}

std::vector<std::string> FileDatabase::listVideos() const
{
    std::vector<std::string> videos;
    for (auto i : fs::directory_iterator(databaseRoot))
    {
        if (fs::is_directory(i.path()))
        {
            videos.push_back(i.path().filename().string());
        }
    }
    return videos;
}

std::optional<DatabaseVideo> FileDatabase::loadVideo(const std::string &key) const
{
    if (!fs::exists(databaseRoot / key / "frames"))
    {
        return std::nullopt;
    }

    return DatabaseVideo{*this, key};
}

bool FileDatabase::saveVocab(const ContainerVocab &vocab, const std::string &key)
{
    cv::Mat myvocab;
    cv::FileStorage fs(to_string(databaseRoot / key), cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab.descriptors();
    return true;
}

std::optional<ContainerVocab> FileDatabase::loadVocab(const std::string &key) const
{
    if (!fs::exists(databaseRoot / key))
    {
        return std::nullopt;
    }
    cv::Mat myvocab;
    cv::FileStorage fs(to_string(databaseRoot / key), cv::FileStorage::READ);
    fs["Vocabulary"] >> myvocab;
    return ContainerVocab{myvocab};
}

DatabaseMetadata FileDatabase::loadMetadata() const
{
    ifstream frame(databaseRoot / Vocab<Frame>::vocab_name);
    ifstream scene(databaseRoot / Vocab<SerializableScene>::vocab_name);
    return {
        getHash(frame),
        getHash(scene),
        config.threshold};
}

VideoMetadata DatabaseVideo::loadMetadata() const
{
    std::ifstream stream(db.databaseRoot / name / VIDEO_METADATA_FILENAME);
    VideoMetadata data{};

    stream.read((char *)&data, sizeof(data));

    return data;
}

QueryVideo make_query_adapter(const SIFTVideo &video, const FileDatabase &db)
{
    auto threshold = db.getConfig().threshold;
    if (threshold == -1)
    {
        throw std::runtime_error("no min_scenes provided");
    }

    auto vocab = loadVocabulary<Vocab<SerializableScene>>(db);

    if (!vocab)
    {
        throw std::runtime_error("no frame vocab available");
    }

    auto frames = video.frames();

    scene_bag_adapter adapter{scene_detect_cursor{*frames, threshold}, video.frames(), *vocab};

    return QueryVideo{video, std::make_unique<decltype(adapter)>(std::move(adapter))};
}

QueryVideo make_query_adapter(const DatabaseVideo &video)
{
    return QueryVideo{video, video.getScenes()};
}
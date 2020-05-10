#include "database.hpp"
#include "fs_compat.hpp"
#include <memory>
#include "vocabulary.hpp"
#include "matcher.hpp"
#include "video.hpp"
#include "scene_detector.hpp"
#include "filter.hpp"
#include <tbb/pipeline.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <atomic>
#include "concepts.hpp"
#include <future>

#define VIDEO_METADATA_FILENAME "metadata.bin"

using namespace std;
using namespace cv;

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


template <typename Read>
class frame_bag_adapter : public ICursor<Frame>
{
    Read reader;
    BOWExtractor extractor;

public:
    frame_bag_adapter(Read &&r, const Vocab<Frame> &v) : reader(std::move(r)), extractor(v) {}
    frame_bag_adapter(Read &r, const Vocab<Frame> &v) : reader(r), extractor(v) {}

    void skip(unsigned int n) override {
        if constexpr(has_arrow_v<Read>) {
            reader->skip(n);
        } else {
            reader.skip(n);
        }
    }

    std::optional<Frame> read() override
    {
        if constexpr(has_arrow_v<Read>) {
            if (auto val = reader->read()) {
                val->frameDescriptor = getFrameDescriptor(*val, extractor);
                return val;
            }
        } else {
            if (auto val = reader.read()) {
                val->frameDescriptor = getFrameDescriptor(*val, extractor);
                return val;
            }
        }
        return std::nullopt;
    }
};

template <typename SceneRead, typename FrameRead>
class scene_bag_adapter : public ICursor<SerializableScene>
{
    SceneRead scene_reader;
    FrameRead frame_reader;

    Vocab<SerializableScene> vocab;
    size_t f_index = 0;

public:
    scene_bag_adapter(SceneRead &&s, FrameRead &&f, const Vocab<SerializableScene> &v) : scene_reader(std::move(s)), frame_reader(std::move(f)), vocab(v) {}
    scene_bag_adapter(SceneRead &s, FrameRead &f, const Vocab<SerializableScene> &v) : scene_reader(s), frame_reader(f), vocab(v) {}
    scene_bag_adapter(SceneRead &s, FrameRead &&f, const Vocab<SerializableScene> &v) : scene_reader(s), frame_reader(std::move(f)), vocab(v) {}
    scene_bag_adapter(SceneRead &&s, FrameRead &f, const Vocab<SerializableScene> &v) : scene_reader(std::move(s)), frame_reader(f), vocab(v) {}

    void skip(unsigned int n) override {
        if constexpr(has_arrow_v<SceneRead>) {
            scene_reader->skip(n);
        } else {
            scene_reader.skip(n);
        }
    }

    std::optional<SerializableScene> read() override
    {
        std::optional<SerializableScene> val;
        if constexpr(has_arrow_v<SceneRead>) {
            val = scene_reader->read();
        } else {
            val = scene_reader.read();
        }
        if (val) {
            std::vector<cv::Mat> frames;
            if constexpr(has_arrow_v<FrameRead>) {
                frame_reader->skip(val->startIdx - f_index);
            } else {
                frame_reader.skip(val->startIdx - f_index);
            }
            

            f_index = val->startIdx;

            while(f_index < val->endIdx) {
                if constexpr(has_arrow_v<FrameRead>) {
                    auto val = frame_reader->read();
                    if constexpr(std::is_same_v<typename decltype(val)::value_type, Frame>) {
                        frames.push_back(val->frameDescriptor);
                    } else {
                        frames.push_back(*val);
                    }
                } else {
                    auto val = frame_reader.read();
                    if constexpr(std::is_same_v<typename decltype(val)::value_type, Frame>) {
                        frames.push_back(val->frameDescriptor);
                    } else {
                        frames.push_back(*val);
                    }
                }
                f_index++;
            }

            std::cout << "bagging scene of length: " << frames.size() << std::endl;
            val->frameBag = baggify(frames.begin(), frames.end(), BOWExtractor{vocab});
            return val;
        }
        return std::nullopt;
    }
};

class scene_detect_cursor : public ICursor<SerializableScene>
{
    std::vector<std::pair<unsigned int, unsigned int>> scenes;
    decltype(scenes.begin()) iterator;

public:
    scene_detect_cursor() : iterator(scenes.begin()) {}

    template <typename Read> 
    scene_detect_cursor(Read&& reader, unsigned int min_scenes)
    {
        scenes = hierarchicalScenes(get_distances(reader, ColorComparator2D{}), min_scenes);
        std::cout << "Detected scenes: " << scenes.size() << std::endl;
        iterator = scenes.begin();
    }

    std::optional<SerializableScene> read() override
    {
        if (iterator == scenes.end())
        {
            return std::nullopt;
        }
        return SerializableScene{*iterator++};
    }

    void skip(unsigned int n) override {
        iterator += n;
    }
};


class CaptureSource : public ICursor<cv::UMat>
{
    VideoCapture cap;

public:
    CaptureSource(const std::string &filename) : cap(filename) {}

    std::optional<cv::UMat> read() override
    {
        UMat image;

        if (cap.read(image))
        {
            return image;
        }

        return std::nullopt;
    }

    void skip(unsigned int n) override {
        for(unsigned int i = 0; i < n; i++) cap.grab();
    }
};

class ColorSource : public ICursor<cv::Mat> {
    CaptureSource source;
    ScaleImage scale;
    Extract2DColorHistogram color;
    size_t counter = 0;

public:
    ColorSource(const std::string &filename, std::pair<int, int> cropsize) 
            : source(filename), scale(cropsize){};

    std::optional<cv::Mat> read() override
    {
        if (auto image = source.read())
        {
            if(counter++ % 40 == 0) {
                std::cout << "video frame: " << counter - 1 << std::endl;
            }
            
            return color(scale(*image));
        }
        return std::nullopt;
    }

    void skip(unsigned int n) override {
        source.skip(n);
    }
};

class FrameSource : public ICursor<Frame>
{
    CaptureSource source;
    std::function<void(UMat, Frame)> callback;
    ScaleImage scale;
    Extract2DColorHistogram color;
    ExtractSIFT sift;
    size_t counter = 0;

public:
    FrameSource(const std::string &filename, std::function<void(UMat, Frame)> callback,
                std::pair<int, int> cropsize) : source(filename), callback(callback), scale(cropsize){};

    std::optional<Frame> read() override
    {
        if (auto image = source.read())
        {
            if(counter++ % 40 == 0) {
                std::cout << "video frame: " << counter - 1 << std::endl;
            }
            
            auto scaled = scale(*image);
            
            if (callback) {
                auto frame = sift.withKeyPoints(scaled);
                frame.colorHistogram = color(scaled);
                callback(*image, frame);
                return frame;
            }
            
            auto colorHistogram = color(scaled);
            auto descriptors = sift(scaled);
            return Frame{descriptors, cv::Mat(), colorHistogram};
        }
        return std::nullopt;
    }
    void skip(unsigned int n) override {
        source.skip(n);
    }
};

std::unique_ptr<ICursor<cv::UMat>> SIFTVideo::images() const
{
    return std::make_unique<CaptureSource>(filename);
}

std::unique_ptr<ICursor<cv::Mat>> SIFTVideo::color() const
{
    return std::make_unique<ColorSource>(filename, cropsize);
}

std::unique_ptr<ICursor<Frame>> SIFTVideo::frames() const
{
    return std::make_unique<FrameSource>(filename, callback, cropsize);
}

std::unique_ptr<ICursor<Frame>> SIFTVideo::frames(const Vocab<Frame>& vocab) const {
    frame_bag_adapter source{FrameSource{filename, callback, cropsize}, vocab};
    return std::make_unique<decltype(source)>(std::move(source));
}

SIFTVideo::SIFTVideo(const std::string &filename, std::function<void(cv::UMat, Frame)> callback, std::pair<int, int> cropsize)
    : IVideo(fs::path(filename).filename()), filename(filename), callback(callback), cropsize(cropsize) {}

SIFTVideo getSIFTVideo(const std::string &filepath, std::function<void(UMat, Frame)> callback, std::pair<int, int> cropsize)
{
    return {filepath, callback, cropsize};
}

std::unique_ptr<ICursor<Frame>> DatabaseVideo::frames() const
{
    auto frame_source = make_frame_source(db.getFileLoader(), name);
    auto vocab = loadVocabulary<Frame>(db);

    if (db.loadStrategy == Eager &&
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
    auto vocab = loadVocabulary<SerializableScene>(db);
    if (db.loadStrategy == Eager &&
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

template<typename T>
auto make_cursor_source(std::unique_ptr<ICursor<T>> source)
{
    size_t index = 0;
    return [source = std::move(source), index](tbb::flow_control &fc) mutable {
        if (auto val = source->read())
        {
            return ordered_adapter<T, size_t>{index++, *val};
        }
        fc.stop();
        return ordered_adapter<T, size_t>{};
    };
}

template<typename Read>
auto make_cursor_source(Read&& source)
{
    size_t index = 0;
    return [source = std::forward<Read>(source), index](tbb::flow_control &fc) mutable {
        auto val = source.read();
        if (val)
        {
            return ordered_adapter<typename decltype(val)::value_type, size_t>{index++, *val};
        }
        fc.stop();
        return ordered_adapter<typename decltype(val)::value_type, size_t>{};
    };
}

void writeMetadata(const VideoMetadata &data, const fs::path &videoDir)
{
    ofstream stream(videoDir / VIDEO_METADATA_FILENAME);
    stream.write((char *)&data, sizeof(data));
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const SIFTVideo &video)
{
    VideoMetadata metadata{};
    fs::path video_dir{databaseRoot / video.name};
    loader.initVideoDir(video.name);
    loader.clearFrames(video.name);

    auto source = make_cursor_source(video.images());

    ScaleImage scale(video.cropsize);
    ExtractSIFT sift;
    Extract2DColorHistogram color;
    SaveFrameSink saveFrame(video.name, getFileLoader());

    std::atomic<size_t> frameCount = 0;

    tbb::parallel_pipeline(16,
        tbb::make_filter<void, ordered_umat>(tbb::filter::serial_out_of_order, [&](tbb::flow_control& fc){
            return source(fc);
        }) &
        tbb::make_filter<ordered_umat, ordered_umat>(tbb::filter::parallel, scale) & 
        tbb::make_filter<ordered_umat, std::pair<Frame, ordered_umat>>(tbb::filter::parallel, [&](auto mat) {
            Frame f;
            f.descriptors = sift(mat).data;
            return make_pair(f, mat);
        }) &
        tbb::make_filter<std::pair<Frame, ordered_umat>, ordered_frame>(tbb::filter::parallel, [&](auto pair) {
            pair.first.colorHistogram = color(pair.second).data;
            return ordered_frame{pair.second.rank, pair.first};
        }) &
        tbb::make_filter<ordered_frame, void>(tbb::filter::parallel, [&](auto frame) {
            frameCount.fetch_add(1, std::memory_order_relaxed);
            saveFrame(frame);
        }));

    metadata.frameCount = frameCount.load(std::memory_order_relaxed);
    writeMetadata(metadata, video_dir);
    return saveVideo(DatabaseVideo{*this, video.name});
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const DatabaseVideo &video)
{
    auto vocab = loadVocabulary<Frame>(*this);
    auto sceneVocab = loadVocabulary<SerializableScene>(*this);
    auto metadata = video.loadMetadata();
    VideoMetadata saveMetadata{metadata};

    auto db_metadata = loadMetadata();

    const bool willExtractScenes = strategy->shouldBaggifyScenes()
        && (metadata != db_metadata)
        && sceneVocab;

    std::unique_ptr<ICursor<SerializableScene>> sceneCursor{std::make_unique<NullCursor<SerializableScene>>()};
    std::unique_ptr<ICursor<cv::Mat>> frameCursor{std::make_unique<NullCursor<cv::Mat>>()};

    BOWExtractor extractor{vocab.value_or(cv::Mat())};
    auto source = make_sift_source(loader, video.name);

    cursor_adapter frames{read_adapter{[&, index = 0]() mutable -> std::optional<cv::Mat> {
        auto frame = source.read();
        if(index++ % 40 == 0) {
            std::cout << "bag frame: " << index - 1 << std::endl;
        }
        if(frame) {
            auto computed = baggify(*frame, extractor);
            if(willExtractScenes)
                loader.saveFrameBag(video.name, index - 1, computed);
            return computed;
        }
        return std::nullopt;
    }}};
    
    if(willExtractScenes) {
        auto scene = make_scene_source(loader, video.name);
        sceneCursor = std::make_unique<decltype(scene)>(std::move(scene));
        auto source = make_frame_bag_source(loader, video.name);
        frameCursor = std::make_unique<decltype(source)>(source);
    }

    scene_detect_cursor scenes{make_color_source(loader, video.name), 
        static_cast<unsigned int>(config.threshold)};

    if (strategy->shouldBaggifyFrames() 
        && metadata.frameHash != db_metadata.frameHash 
        && vocab)
    {
        saveMetadata.frameHash = db_metadata.frameHash;

        frameCursor = std::make_unique<decltype(frames)>(std::move(frames));
    }
    if (strategy->shouldComputeScenes() 
        && metadata.threshold != db_metadata.threshold
        && config.threshold != -1)
    {
        saveMetadata.threshold = config.threshold;
        loader.clearScenes(video.name);

        sceneCursor = std::make_unique<decltype(scenes)>(std::move(scenes));
    }
    if (willExtractScenes)
    {
        saveMetadata.sceneHash = db_metadata.sceneHash;
        size_t index = 0;
        scene_bag_adapter adapter{std::move(sceneCursor), std::move(frameCursor), *sceneVocab};
        while(auto scene = adapter.read()) loader.saveScene(video.name, index++, *scene);
    } else {
        auto writeScenes = std::async(std::launch::async, [&](){
            size_t index = 0;
            while(auto scene = sceneCursor->read()) loader.saveScene(video.name, index++, *scene);
        });
        auto writeFrames = std::async(std::launch::async, [&](){
            size_t index = 0;
            while(auto frame = frameCursor->read()) loader.saveFrameBag(video.name, index++, *frame);
        });

        writeScenes.wait();
        writeFrames.wait();
    }

    writeMetadata(saveMetadata, databaseRoot / video.name);
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
    if (!fs::exists(databaseRoot / key))
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

    auto frame_vocab = loadVocabulary<Frame>(db);
    auto scene_vocab = loadVocabulary<SerializableScene>(db);

    if (!frame_vocab)
    {
        throw std::runtime_error("no frame vocab available");
    }

    if (!scene_vocab)
    {
        throw std::runtime_error("no scene vocab available");
    }

    scene_bag_adapter adapter{
        scene_detect_cursor{*video.color(), static_cast<unsigned int>(threshold)}, 
        frame_bag_adapter{video.frames(), *frame_vocab}, *scene_vocab};

    return QueryVideo{video, std::make_unique<decltype(adapter)>(std::move(adapter))};
}

QueryVideo make_query_adapter(const DatabaseVideo &video)
{
    return QueryVideo{video, video.getScenes()};
}
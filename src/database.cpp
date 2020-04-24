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

#define HBINS 32
#define SBINS 30

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

/* creates folder if it doesn't exist, otherwise throws an exception */
void createFolder(const string &folder_name)
{
    fs::create_directories(fs::current_path() / folder_name);
}

string getAlphas(const string &input)
{
#ifdef CLEAN_NAMES
    // TODO: check for at least one alpha char
    string output;
    copy_if(input.begin(), input.end(), back_inserter(output), [](auto c) -> bool { return isalnum(c); });
    return output;
#else
    return input;
#endif
}

std::string to_string(const fs::path &path)
{
    return path.string();
}

template <typename Range>
void writeSequential(ofstream &fs, const Range &range)
{
    size_t length = range.size();
    fs.write((char *)&length, sizeof(length));
    for (const auto &val : range)
    {
        fs.write((char *)&val, sizeof(val));
    }
}

template <typename T>
std::vector<T> readSequence(ifstream &fs)
{
    size_t length = 0;
    fs.read((char *)&length, sizeof(length));

    std::vector<T> items(length);
    fs.read((char *)&items[0], sizeof(T) * length);

    return items;
}

cv::Mat readMat(ifstream &fs)
{
    int rows, cols, type, channels;
    fs.read((char *)&rows, sizeof(int));     // rows
    fs.read((char *)&cols, sizeof(int));     // cols
    fs.read((char *)&type, sizeof(int));     // type
    fs.read((char *)&channels, sizeof(int)); // channels

    Mat mat(rows, cols, type);
    for (int r = 0; r < rows; r++)
    {
        fs.read((char *)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    return mat;
}

void writeMat(const cv::Mat &mat, ofstream &fs)
{
    int type = mat.type();
    int channels = mat.channels();
    fs.write((char *)&mat.rows, sizeof(int)); // rows
    fs.write((char *)&mat.cols, sizeof(int)); // cols
    fs.write((char *)&type, sizeof(int));     // type
    fs.write((char *)&channels, sizeof(int)); // channels

    // Data
    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
    }
    else
    {
        int rowsz = CV_ELEM_SIZE(type) * mat.cols;
        for (int r = 0; r < mat.rows; ++r)
        {
            fs.write(mat.ptr<char>(r), rowsz);
        }
    }
}

SerializableScene SceneRead(const std::string &filename)
{
    size_t startIdx = 0, endIdx = 0;

    ifstream fs(filename, fstream::binary);
    fs.read((char *)&startIdx, sizeof(startIdx));
    fs.read((char *)&endIdx, sizeof(endIdx));

    return {readMat(fs), startIdx, endIdx};
}

bool SceneWrite(const std::string &filename, const SerializableScene &scene)
{
    ofstream fs(filename, fstream::binary);
    if (!fs.is_open())
    {
        return false;
    }

    fs.write((char *)&scene.startIdx, sizeof(scene.startIdx));
    fs.write((char *)&scene.endIdx, sizeof(scene.endIdx));

    writeMat(scene.frameBag, fs);
    return true;
}

bool SIFTwrite(const string &filename, const Frame &frame)
{
    ofstream fs(filename, fstream::binary);
    if (!fs.is_open())
    {
        return false;
    }

    writeMat(frame.descriptors, fs);
    writeSequential(fs, frame.keyPoints);
    writeMat(frame.frameDescriptor, fs);
    writeMat(frame.colorHistogram, fs);

    return true;
}

Frame SIFTread(const string &filename)
{
    ifstream fs(filename, fstream::binary);

    auto mat = readMat(fs);

    auto keyPoints = readSequence<KeyPoint>(fs);

    auto frameMat = readMat(fs);
    auto colorHistogram = readMat(fs);

    return {keyPoints, mat, frameMat, colorHistogram};
}

class CaptureSource : public ICursor<cv::UMat> {
    VideoCapture cap;

public:
    CaptureSource(const std::string& filename) : cap(filename, cv::CAP_ANY) {}

    std::optional<cv::UMat> read() override {
        UMat image;

        if(cap.read(image))
        {
            return image;
        }

        return std::nullopt;
    }
};

class FrameSource : public ICursor<Frame> {
    CaptureSource source;
    std::function<void(UMat, Frame)> callback;
    ScaleImage scale;
    ExtractColorHistogram color;
    ExtractSIFT sift;

public:
    FrameSource(const std::string& filename, std::function<void(UMat, Frame)> callback,
        std::pair<int, int> cropsize) : source(filename), callback(callback), scale(cropsize) {};

    std::optional<Frame> read() override {
        if(auto image = source.read()) {
            auto ordered = ordered_umat{0, *image};
            auto scaled = scale(ordered);
            auto colorHistogram = color(scaled);
            auto descriptors = sift(scaled);
            Frame frame{descriptors.data, cv::Mat(), colorHistogram.data};
            if(callback)
                callback(*image, frame);

            return frame;
        }
        return std::nullopt;
    }
};

std::unique_ptr<ICursor<cv::UMat>> SIFTVideo::images() const {
    return std::make_unique<CaptureSource>(filename);
}

SIFTVideo getSIFTVideo(const std::string &filepath, std::function<void(UMat, Frame)> callback, std::pair<int, int> cropsize)
{
    return {filepath, callback, cropsize};
}

const std::string SerializableScene::vocab_name = "SceneVocab.mat";
const std::string Frame::vocab_name = "FrameVocab.mat";

template <typename T>
const std::string Vocab<T>::vocab_name = T::vocab_name;

FileDatabase::FileDatabase(const std::string &databasePath,
                           std::unique_ptr<IVideoStorageStrategy> &&strat, StrategyType l, RuntimeArguments args)
    : strategy(std::move(strat)), loadStrategy(l), config(args, strategy->getType(), l), databaseRoot(databasePath),
      loader(databasePath)
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

auto make_image_source(std::unique_ptr<ICursor<UMat>> source) {
    size_t index = 0;
    return [&source, index](tbb::flow_control& fc) mutable {
        if(auto image = source->read()) {
            return ordered_umat{index++, *image};
        }
        fc.stop();
        return ordered_umat{};
    };
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const SIFTVideo &video, const::string& key)
{
    fs::path video_dir{databaseRoot / key};
    loader.initVideoDir(key);
    loader.clearFrames(key);

    auto source = make_image_source(video.images());

    ScaleImage scale(video.cropsize);
    ExtractSIFT sift;
    ExtractColorHistogram color;
    SaveFrameSink saveFrame(key, getFileLoader());

    auto filter = tbb::make_filter<void, ordered_umat>(tbb::filter::serial_out_of_order, [&source](auto fc){
            return source(fc);
        }) &
        tbb::make_filter<ordered_umat, ordered_umat>(tbb::filter::parallel, scale) &
        tbb::make_filter<ordered_umat, std::pair<Frame, ordered_umat>>(tbb::filter::parallel, [&](auto mat){
            Frame f;
            f.descriptors = sift(mat).data;
            return make_pair(f, mat);
        }) &
        tbb::make_filter<std::pair<Frame, ordered_umat>, ordered_frame>(tbb::filter::parallel, [&](auto pair) {
            pair.first.colorHistogram = color(pair.second).data;
            return ordered_frame{pair.second.rank, pair.first};
        });

    if(strategy->shouldBaggifyFrames()) {
        if(auto vocab = loadVocabulary<Vocab<Frame>>(*this)) {
            ExtractFrame frame(*vocab);
            filter = filter & tbb::make_filter<ordered_frame, ordered_frame>(tbb::filter::parallel, [frame](auto f){
                f.data.frameDescriptor = frame(f.data.descriptors);
                return f;
            });
        }
    }

    tbb::parallel_pipeline(300,
        filter &
        tbb::make_filter<ordered_frame, void>(tbb::filter::parallel, saveFrame) 
    );

    return saveVideo(DatabaseVideo{*this, databaseRoot / key, key});
}

std::optional<DatabaseVideo> FileDatabase::saveVideo(const DatabaseVideo& video) {
    auto vocab = loadVocabulary<Vocab<Frame>>(*this);
    auto sceneVocab = loadVocabulary<Vocab<SerializableScene>>(*this);
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

    return DatabaseVideo{*this, databaseRoot / key, key};
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

std::optional<Frame> FileLoader::readFrame(const std::string &videoName, size_t index) const
{
    auto def = cv::Mat();
    ifstream stream(rootDir / videoName / "frames" / (to_string(index) + ".keypoints"), fstream::binary);

    auto features = readFrameFeatures(videoName, index);
    auto bag = readFrameBag(videoName, index);
    auto color = readFrameColorHistogram(videoName, index);
    if (!(stream.is_open() || bag || features || color))
    {
        return std::nullopt;
    }

    auto keyPoints = readSequence<KeyPoint>(stream);

    return Frame{
        keyPoints,
        features.value_or(def),
        bag.value_or(def),
        color.value_or(def)};
}

std::optional<cv::Mat> FileLoader::readFrameData(const std::string &videoName, const std::string &fileName) const
{
    ifstream stream(rootDir / videoName / "frames" / fileName, fstream::binary);
    if (!stream.is_open())
    {
        return nullopt;
    }

    return readMat(stream);
}

std::optional<cv::Mat> FileLoader::readFrameFeatures(const std::string &videoName, size_t index) const
{
    return readFrameData(videoName, to_string(index) + ".sift");
}
std::optional<cv::Mat> FileLoader::readFrameColorHistogram(const std::string &videoName, size_t index) const
{
    return readFrameData(videoName, to_string(index) + ".color");
}
std::optional<cv::Mat> FileLoader::readFrameBag(const std::string &videoName, size_t index) const
{
    return readFrameData(videoName, to_string(index) + ".bag");
}

std::optional<SerializableScene> FileLoader::readScene(const std::string &videoName, size_t index) const
{
    auto path = rootDir / videoName / "scenes" / to_string(index);
    if (!fs::exists(path))
    {
        return std::nullopt;
    }

    return SceneRead(path.string());
}

bool FileLoader::saveFrame(const std::string &video, size_t index, const Frame &frame) const
{
    ofstream stream(rootDir / video / "frames" / (to_string(index) + ".keypoints"), fstream::binary);
    if (!stream.is_open())
    {
        return false;
    }
    writeSequential(stream, frame.keyPoints);

    return saveFrameFeatures(video, index, frame.descriptors) &&
           saveFrameColorHistogram(video, index, frame.colorHistogram) &&
           saveFrameBag(video, index, frame.frameDescriptor);
}

bool FileLoader::saveFrameData(const std::string &videoName, const std::string &filename, const cv::Mat &mat) const
{
    ofstream stream(rootDir / videoName / "frames" / filename, fstream::binary);
    if (!stream.is_open())
    {
        return false;
    }

    writeMat(mat, stream);
    return true;
}

bool FileLoader::saveFrameFeatures(const std::string &videoName, size_t index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".sift", mat);
}

bool FileLoader::saveFrameColorHistogram(const std::string &videoName, size_t index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".color", mat);
}

bool FileLoader::saveFrameBag(const std::string &videoName, size_t index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".bag", mat);
}

bool FileLoader::saveScene(const std::string &video, size_t index, const SerializableScene &scene) const
{
    SceneWrite(to_string(rootDir / video / "scenes" / std::to_string(index)), scene);
    return true;
}

void FileLoader::initVideoDir(const std::string &video) const
{
    fs::create_directories(rootDir / video / "frames");
    fs::create_directories(rootDir / video / "scenes");
}

void FileLoader::clearFrames(const std::string &video) const
{
    clearDir(rootDir / video / "frames");
}

void FileLoader::clearScenes(const std::string &video) const
{
    clearDir(rootDir / video / "scenes");
}

void clearDir(fs::path path)
{
    if (fs::exists(path))
    {
        fs::remove_all(path);
    }

    fs::create_directories(path);
}

double ColorComparator::operator()(const Frame &f1, const Frame &f2) const
{
    return operator()(f1.colorHistogram, f2.colorHistogram);
}
double ColorComparator::operator()(const cv::Mat &f1, const cv::Mat &f2) const
{
    if (f1.rows != HBINS || f1.cols != SBINS)
    {
        std::cerr
            << "rows: " << f1.rows
            << " cols: " << f1.cols << std::endl;
        throw std::runtime_error("color histogram is wrong size");
    }

    if (f1.size() != f2.size())
    {
        throw std::runtime_error("colorHistograms not matching");
    }

    auto subbed = f1 - f2;
    auto val = cv::sum(subbed)[0];
    return std::abs(val);
}

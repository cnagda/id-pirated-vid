#include "database.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <cctype>
#include <experimental/filesystem>
#include <memory>
#include "vocabulary.hpp"
#include "matcher.hpp"
#include "video.hpp"
#include "scene_detector.hpp"
#include <boost/range/algorithm.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include "imgproc.hpp"
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
    v_size startIdx = 0, endIdx = 0;

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

SIFTVideo getSIFTVideo(const std::string &filepath, std::function<void(UMat, Frame)> callback, std::pair<int, int> cropsize)
{
    vector<Frame> frames;

    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

    VideoCapture cap(filepath, CAP_ANY);

    vector<KeyPoint> keyPoints;
    UMat image;

    size_t index = 0;

    auto num_frames = cap.get(CAP_PROP_FRAME_COUNT);

    while (cap.read(image))
    { // test only loading 2 frames
        if (!(++index % 40))
        {
            std::cout << "Frame " << index << "/" << num_frames << std::endl;
        }
        UMat descriptors, colorHistogram, hsv;

        image = scaleToTarget(image, cropsize.first, cropsize.second);
        cvtColor(image, hsv, COLOR_BGR2HSV);

        detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);

        std::vector<int> histSize{HBINS, SBINS};
        // hue varies from 0 to 179, see cvtColor
        std::vector<float> ranges{0, 180, 0, 256};
        // we compute the histogram from the 0-th and 1-st channels
        std::vector<int> channels{0, 1};

        calcHist(std::vector<decltype(hsv)>{hsv}, channels, Mat(), // do not use mask
                 colorHistogram, histSize, ranges,
                 true);
        normalize(colorHistogram, colorHistogram, 0, 1, NORM_MINMAX, -1, Mat());

        Mat c, d;
        colorHistogram.copyTo(c);
        descriptors.copyTo(d);

        Frame frame{keyPoints, d, Mat(), c};

        frames.push_back(frame);

        if (callback)
            callback(image, frame);
    }

    return SIFTVideo(frames);
}

const std::string SerializableScene::vocab_name = "SceneVocab.mat";
const std::string Frame::vocab_name = "FrameVocab.mat";

template <typename T>
const std::string Vocab<T>::vocab_name = T::vocab_name;

FileDatabase::FileDatabase(const std::string &databasePath,
                           std::unique_ptr<IVideoStorageStrategy> &&strat, LoadStrategy l, RuntimeArguments args)
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

std::optional<DatabaseVideo> FileDatabase::saveVideo(IVideo &video)
{
    fs::path video_dir{databaseRoot / video.name};
    loader.initVideoDir(video.name);

    auto &frames = video.frames();
    auto &scenes = video.getScenes();
    std::vector<SerializableScene> loadedScenes;

    if (strategy->shouldBaggifyFrames(video))
    {
        if (auto vocab = loadVocabulary<Vocab<Frame>>(*this); vocab)
        {
            for (auto &frame : frames)
            {
                loadFrameDescriptor(frame, vocab->descriptors());
            }
        }
    }

    if (!frames.empty())
    {
        loader.clearFrames(video.name);
    }

    loader.saveRange(frames, video.name, 0);

    if (!scenes.empty())
    {
        loader.clearScenes(video.name);
        boost::copy(scenes, std::back_inserter(loadedScenes));
    }
    else if(strategy->shouldComputeScenes(video)) {
        //auto flat = convolutionalDetector(video, ColorComparator{}, config.threshold);
        auto distances = get_distances(make_distance_source(loader, video.name), ColorComparator{});
        auto flat = hierarchicalScenes(distances, config.threshold);

        if (!flat.empty())
        {
            loader.clearScenes(video.name);
            boost::copy(flat | boost::adaptors::transformed([](auto pair) {
                            return SerializableScene{pair.first, pair.second};
                        }),
                        std::back_inserter(loadedScenes));
        }
    }

    if (strategy->shouldBaggifyScenes(video))
    {
        for (auto &scene : loadedScenes)
        {
            loadSceneDescriptor(scene, video, *this);
        }
    }

    loader.saveRange(loadedScenes, video.name, 0);

    return DatabaseVideo{*this, video.name, frames, loadedScenes};
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

    std::vector<Frame> frames;
    v_size index = 0;

    if (loadStrategy == AggressiveLoadStrategy{})
    {
        while (auto f = loader.readFrame(key, index++))
            frames.push_back(f.value());
    }

    if (!fs::exists(databaseRoot / key / "scenes"))
    {
        return DatabaseVideo{*this, key, frames};
    }

    std::vector<SerializableScene> scenes;
    index = 0;
    if (loadStrategy == AggressiveLoadStrategy{})
    {
        while (auto f = loader.readScene(key, index++))
            scenes.push_back(f.value());
    }

    return DatabaseVideo{*this, key, frames, scenes};
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

std::vector<SerializableScene> &DatabaseVideo::getScenes() &
{
    if (sceneCache.empty())
    {
        auto loader = db.getFileLoader();
        v_size index = 0;
        while (auto scene = loader.readScene(name, index++))
            sceneCache.push_back(scene.value());

        if (sceneCache.empty())
        {
            auto config = db.getConfig();
            if (config.threshold == -1)
            {
                throw std::runtime_error("no threshold was provided to calculate scenes");
            }

            //auto ss = convolutionalDetector(*this, ColorComparator{}, config.threshold);
            auto distances = get_distances(make_distance_source(loader, name), ColorComparator{});
            auto ss = hierarchicalScenes(distances, config.threshold);


            std::cout << "Found " << ss.size() << " scenes, serializing now" << std::endl;
            boost::push_back(sceneCache, ss | boost::adaptors::transformed([index = 0](auto scene) {
                                             return SerializableScene{scene.first, scene.second};
                                         }));
        }
    }

    return sceneCache;
}

std::vector<Frame> &DatabaseVideo::frames() &
{
    if (frameCache.empty())
    {
        auto loader = db.getFileLoader();
        IVideo::size_type index = 0;
        while (auto frame = loader.readFrame(name, index++))
            frameCache.push_back(frame.value());
    }
    return frameCache;
}

std::optional<Frame> FileLoader::readFrame(const std::string &videoName, v_size index) const
{
    auto def = cv::Mat();
    ifstream stream(rootDir / videoName / "frames" / (to_string(index) + ".keypoints"), fstream::binary);

    auto features = readFrameFeatures(videoName, index);
    auto bag = readFrameBag(videoName, index);
    auto color = readFrameColorHistogram(videoName, index);
    if(!(stream.is_open() || bag || features || color)) {
        return std::nullopt;
    }

    auto keyPoints = readSequence<KeyPoint>(stream);

    return Frame{
        keyPoints,
        features.value_or(def),
        bag.value_or(def),
        color.value_or(def)
    };
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

std::optional<cv::Mat> FileLoader::readFrameFeatures(const std::string &videoName, v_size index) const
{
    return readFrameData(videoName, to_string(index) + ".sift");
}
std::optional<cv::Mat> FileLoader::readFrameColorHistogram(const std::string &videoName, v_size index) const
{
    return readFrameData(videoName, to_string(index) + ".color");
}
std::optional<cv::Mat> FileLoader::readFrameBag(const std::string &videoName, v_size index) const
{
    return readFrameData(videoName, to_string(index) + ".bag");
}

std::optional<SerializableScene> FileLoader::readScene(const std::string &videoName, v_size index) const
{
    auto path = rootDir / videoName / "scenes" / to_string(index);
    if (!fs::exists(path))
    {
        return std::nullopt;
    }

    return SceneRead(path.string());
}

bool FileLoader::saveFrame(const std::string &video, v_size index, const Frame &frame) const
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

bool FileLoader::saveFrameFeatures(const std::string &videoName, v_size index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".sift", mat);
}

bool FileLoader::saveFrameColorHistogram(const std::string &videoName, v_size index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".color", mat);
}

bool FileLoader::saveFrameBag(const std::string &videoName, v_size index, const cv::Mat &mat) const
{
    return saveFrameData(videoName, to_string(index) + ".bag", mat);
}

bool FileLoader::saveScene(const std::string &video, v_size index, const SerializableScene &scene) const
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

DatabaseVideo make_scene_adapter(FileDatabase &db, IVideo &video, const std::string &key)
{
    std::vector<SerializableScene> loadedScenes;

    auto config = db.getConfig();
    //auto scenes = convolutionalDetector(video, ColorComparator{}, config.threshold);
    auto distances = get_distances(make_distance_source(db.getFileLoader(), key), ColorComparator{});
    auto scenes = hierarchicalScenes(distances, config.threshold);


    std::cout << "Found " << scenes.size() << " scenes, serializing now" << std::endl;

    if (!scenes.empty())
    {
        for (auto &scene : scenes)
        {
            loadedScenes.emplace_back(scene.first, scene.second);
        }
    }

    return {db, key, video.frames(), loadedScenes};
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

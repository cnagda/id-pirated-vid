#include "database.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <cctype>
#include <experimental/filesystem>
#include <memory>
#include "vocabulary.hpp"
#include "matcher.hpp"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

/* creates folder if it doesn't exist, otherwise throws an exception */
void createFolder(const string& folder_name) {
    fs::create_directories(fs::current_path() / folder_name);
}

string getAlphas(const string& input)
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

SerializableScene SceneRead(const std::string& filename) {
    SIFTVideo::size_type startIdx = 0, endIdx = 0;

    ifstream fs(filename, fstream::binary);
    fs.read((char*)&startIdx, sizeof(startIdx));
    fs.read((char*)&endIdx, sizeof(endIdx));

    // Header
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

    return SerializableScene{mat, startIdx, endIdx};
}

void SceneWrite(const std::string& filename, const SerializableScene& scene) {
    ofstream fs(filename, fstream::binary);
    fs.write((char*)&scene.startIdx, sizeof(scene.startIdx));
    fs.write((char*)&scene.endIdx, sizeof(scene.endIdx));

    const auto& mat = scene.frameBag;
    // Header
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

void SIFTwrite(const string &filename, const Frame& frame)
{
    const auto& mat = frame.descriptors;
    const auto& keyPoints = frame.keyPoints;
    ofstream fs(filename, fstream::binary);

    // Header
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

    fs.write((char *)&keyPoints[0], keyPoints.size() * sizeof(KeyPoint));
}

Frame SIFTread(const string &filename)
{
    ifstream fs(filename, fstream::binary);

    // Header
    int rows, cols, type, channels;
    fs.read((char *)&rows, sizeof(int));     // rows
    fs.read((char *)&cols, sizeof(int));     // cols
    fs.read((char *)&type, sizeof(int));     // type
    fs.read((char *)&channels, sizeof(int)); // channels

    // Data
    Mat mat(rows, cols, type);
    vector<KeyPoint> keyPoints;
    for (int r = 0; r < rows; r++)
    {
        fs.read((char *)(mat.data + r * cols * CV_ELEM_SIZE(type)), CV_ELEM_SIZE(type) * cols);
    }

    for (int r = 0; r < rows; r++)
    {
        KeyPoint k;
        fs.read((char *)&k, sizeof(KeyPoint));
        keyPoints.push_back(k);
    }
    return Frame{keyPoints, mat};
}

cv::Mat scaleToTarget(cv::Mat image, int targetWidth, int targetHeight){
    int srcWidth = image.cols;
    int srcHeight = image.rows;

    double ratio = std::min((double)targetHeight/srcHeight, (double)targetWidth/srcWidth);

    cv::Mat retval;

    resize(image, retval, Size(), ratio, ratio);
    return retval;
}

SIFTVideo getSIFTVideo(const std::string& filepath, std::function<void(Mat, Frame)> callback, std::pair<int, int> cropsize) {
    vector<Frame> frames;

    Ptr<FeatureDetector> detector = xfeatures2d::SiftFeatureDetector::create(500);

    VideoCapture cap(filepath, CAP_ANY);

    vector<KeyPoint> keyPoints;
    Mat descriptors, image;

    size_t index = 0;

    int num_frames = cap.get(CAP_PROP_FRAME_COUNT);

    while (cap.read(image))
    { // test only loading 2 frames
        if(!(++index % 10)){
            std::cout << "Frame " << index << "/" << num_frames << std::endl;
        }

        image = scaleToTarget(image, cropsize.first, cropsize.second);

        detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);

        Frame frame{keyPoints, descriptors};

        frames.push_back(frame);

        if(callback) callback(image, frame);
    }

    return SIFTVideo(frames);
}

const std::string IScene::vocab_name = "SceneVocab.mat";
const std::string Frame::vocab_name = "FrameVocab.mat";

template<typename T>
const std::string Vocab<T>::vocab_name = T::vocab_name;

std::unique_ptr<IVideo> FileDatabase::saveVideo(IVideo& video) {
    fs::path video_dir(databaseRoot / video.name);
    fs::create_directories(video_dir);

    auto frames = video.frames();
    SIFTVideo::size_type index = 0;
    auto& scenes = video.getScenes();
    std::vector<SerializableScene> loadedScenes;

    if(!frames.empty()) {
        fs::create_directories(video_dir / "frames");
    }
    for(auto& frame : frames) {
        SIFTwrite(video_dir / "frames" / std::to_string(index++), frame);
    }

    if(strategy->shouldBaggifyFrames(video)) {
        auto vocab = loadOrComputeVocab<Vocab<Frame>>(*this, config.KFrames);
    }

    if(!scenes.empty()) {
        for(auto& scene : scenes) {
            SceneWrite(video_dir / "scenes" / std::to_string(index++), *scene);
            loadedScenes.push_back(*scene);
        }
    }
    else if(strategy->shouldComputeScenes(video)) {
        auto vocab = loadOrComputeVocab<Vocab<Frame>>(*this, config.KFrames);

        auto comp = BOWComparator(vocab.descriptors());
        auto scenes = flatScenes(video, comp, 0.2);

        if(!scenes.empty()) {
            fs::create_directories(video_dir / "scenes");
            SIFTVideo::size_type index = 0;
            for(auto& scene : scenes) {
                auto s = SerializableScene(scene.first, scene.second);
                SceneWrite(video_dir / "scenes" / std::to_string(index++),
                    s);
                loadedScenes.push_back(s);
            }

            if(strategy->shouldBaggifyScenes(video)) {
                auto frameVocab = loadOrComputeVocab<Vocab<IScene>>(*this, config.KScenes);
                SIFTVideo::size_type index = 0;
                for(auto& scene : loadedScenes) {
                    auto access = [&vocab](auto frame){ return baggify(frame.descriptors, vocab.descriptors()); };
                    auto rng = scene.getFrameRange(video) | boost::adaptors::transformed(access);
                    scene.frameBag = baggify(rng.begin(), rng.end(), frameVocab.descriptors());
                    SceneWrite(video_dir / "scenes" / std::to_string(index++), scene);
                }
            }
        }
    }

    return std::make_unique<DatabaseVideo>(*this, video.name, frames, loadedScenes);
}

std::vector<std::unique_ptr<IVideo>> FileDatabase::loadVideos() const {
    std::vector<std::unique_ptr<IVideo>> videos;
    for(auto i : fs::directory_iterator(databaseRoot)) {
        if(fs::is_directory(i.path())) {
            auto v = loadVideo(i.path().filename());
            std::move(v.begin(), v.end(), std::back_inserter(videos));
        }
    }
    return videos;
}

std::vector<std::unique_ptr<IVideo>> FileDatabase::loadVideo(const std::string& key) const {
    if(key == "") {
        return loadVideos();
    }

    std::vector<std::unique_ptr<IVideo>> vid;

    if(!fs::exists(databaseRoot / key / "frames")) {
        return vid;
    }

    std::vector<Frame> frames;
    SIFTVideo::size_type index = 0;

    while(auto f = loader.readFrame(key, index++)) frames.push_back(f.value());

    if(!fs::exists(databaseRoot / key / "scenes")) {
        vid.push_back(std::make_unique<DatabaseVideo>(*this, key, frames));
        return vid;
    }

    std::vector<SerializableScene> scenes;
    index = 0;
    while(auto f = loader.readScene(key, index++)) scenes.push_back(f.value());

    vid.push_back(std::make_unique<DatabaseVideo>(*this, key, frames, scenes));
    return vid;
}

bool FileDatabase::saveVocab(const ContainerVocab& vocab, const std::string& key) {
    cv::Mat myvocab;
    cv::FileStorage fs(databaseRoot / key, cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab.descriptors();
    return true;
}

std::optional<ContainerVocab> FileDatabase::loadVocab(const std::string& key) const {
    if(!fs::exists(databaseRoot / key)) {
        return std::nullopt;
    }
    cv::Mat myvocab;
    cv::FileStorage fs(databaseRoot / key, cv::FileStorage::READ);
    fs["Vocabulary"] >> myvocab;
    return std::make_optional<ContainerVocab>(myvocab);
}

std::vector<std::unique_ptr<IScene>>& DatabaseVideo::getScenes() & {
    if(sceneCache.empty()) {
        auto loader = db.getFileLoader();
        SIFTVideo::size_type index = 0;
        while(auto scene = loader.readScene(name, index++))
            sceneCache.push_back(std::make_unique<DatabaseScene>(
                *this, db, scene.value()));

        if(sceneCache.empty()) {
            auto config = db.getConfig();
            auto vocab = loadVocabulary<Vocab<Frame>>(db);
            if(!vocab) {
                throw std::runtime_error("video could not load vocab");
            }
            auto comp = BOWComparator(vocab->descriptors());
            auto ss = flatScenes(*this, comp, config.threshold);
            boost::push_back(sceneCache, ss
            | boost::adaptors::transformed([this](auto scene) {
                auto f = frames();
                return std::make_unique<DatabaseScene>(*this, db,
                std::make_pair(f.begin() + scene.first, f.begin() + scene.second));
            }));
        }
    }

    return sceneCache;
}

std::optional<Frame> FileLoader::readFrame(const std::string& videoName, SIFTVideo::size_type index) const {
    auto path = rootDir / videoName / "frames" / to_string(index);
    if(!fs::exists(path)) {
        return std::nullopt;
    }

    return SIFTread(path);
}
std::optional<SerializableScene> FileLoader::readScene(const std::string& videoName, SIFTVideo::size_type index) const {
    auto path = rootDir / videoName / "scenes" / to_string(index);
    if(!fs::exists(path)) {
        return std::nullopt;
    }

    // unimplemented
    return SceneRead(path);
}

DatabaseVideo make_scene_adapter(FileDatabase& db, IVideo& video, const std::string& key) {
    std::vector<SerializableScene> loadedScenes;

    auto config = db.getConfig();

    auto frames = video.frames();
    auto vocab = loadOrComputeVocab<Vocab<Frame>>(db, config.KFrames);

    auto comp = BOWComparator(vocab.descriptors());
    auto scenes = flatScenes(video, comp, 0.2);

    if(!scenes.empty()) {
        SIFTVideo::size_type index = 0;
        for(auto& scene : scenes) {
            auto s = SerializableScene(scene.first, scene.second);
            loadedScenes.push_back(s);
        }
    }

    return DatabaseVideo(db, key, frames, loadedScenes);
}

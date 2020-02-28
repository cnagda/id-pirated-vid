#include "database.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <cctype>
#include <experimental/filesystem>
#include "keyframes.hpp"
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
    decltype(frames)::size_type index = 0;
    
    if(!frames.empty()) {
        fs::create_directories(video_dir / "frames");
    }
    for(auto& frame : frames) {
        SIFTwrite(video_dir / "frames" / std::to_string(index++), frame);
    }
    
    if(strategy->shouldBaggifyFrames(video)) {
        auto vocab = loadOrComputeVocab<Vocab<Frame>>(*this, args.KFrame);
    }

    if(strategy->shouldComputeScenes(video)) {
        auto vocab = loadOrComputeVocab<Vocab<Frame>>(*this, args.KFrame);

        auto comp = BOWComparator(vocab.descriptors());
        auto scenes = flatScenes(video, comp, 0.2);
    }

    if(strategy->shouldBaggifyScenes(video)) {
        auto vocab = loadOrComputeVocab<Vocab<IScene>>(*this, args.KScenes);
    }

    return std::make_unique<InputVideoAdapter<SIFTVideo>>(video);
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
    std::vector<Frame> frames;

    if(!fs::exists(databaseRoot / key / "frames")) {
        return vid;
    }
    
    auto it = fs::directory_iterator{databaseRoot / key / "frames"};
    vector<fs::directory_entry> files(it, fs::end(it));
    sort(files.begin(), files.end(), [](auto a, auto b) {
        return stoi(a.path().filename()) < stoi(b.path().filename());
    });

    for (auto frame_path : files)
    {
        auto frame = SIFTread(frame_path.path());
        frames.push_back(frame);
    }    

    vid.push_back(std::move(make_unique<InputVideoAdapter<SIFTVideo>>(SIFTVideo(frames), key)));
    return vid;
}

bool FileDatabase::saveVocab(const IVocab& vocab, const std::string& key) {
    cv::Mat myvocab;
    cv::FileStorage fs(databaseRoot / key, cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab.descriptors();
    return true;
}
std::unique_ptr<IVocab> FileDatabase::loadVocab(const std::string& key) const {
    if(!fs::exists(databaseRoot / key)) {
        return nullptr;
    }
    cv::Mat myvocab;
    cv::FileStorage fs(databaseRoot / key, cv::FileStorage::READ);
    fs["Vocabulary"] >> myvocab;
    return std::make_unique<ContainerVocab>(myvocab);
}

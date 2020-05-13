#include "storage.hpp"
#include <opencv2/core/mat.hpp>
#include <fstream>
#include <iterator>
#include <iostream>

using namespace std;
using namespace cv;

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
    int type = 0, dims = 0;
    fs.read((char *)&type, sizeof(type));     // type
    fs.read((char *)&dims, sizeof(dims));     // type

    if(dims <= 2) {
        int rows = 0, cols = 0;
        fs.read((char *)&rows, sizeof(int));     // rows
        fs.read((char *)&cols, sizeof(int));     // cols

        Mat mat(rows, cols, type);
        fs.read(mat.ptr<char>(), mat.total()*mat.elemSize());

        return mat;
    } else {
        std::vector<int> sizes(dims);
        fs.read((char*)&sizes[0], dims * sizeof(int));

        Mat mat(dims, &sizes[0], type);

        fs.read(mat.ptr<char>(), mat.total()*mat.elemSize());

        return mat;
    }
}

void writeMat(const cv::Mat &mat, ofstream &fs)
{
    int type = mat.type(), dims = mat.dims;
    fs.write((char *)&type, sizeof(int));     // type
    fs.write((char *)&dims, sizeof(int));     // type

    if(dims <= 2) {
        fs.write((char *)&mat.rows, sizeof(int)); // rows
        fs.write((char *)&mat.cols, sizeof(int)); // cols
    } else {
        fs.write((char*)mat.size.p, dims * sizeof(int));
    }

    if (mat.isContinuous())
    {
        fs.write(mat.ptr<char>(), mat.total()*mat.elemSize());
    } else {
        for(int r = 0; r < mat.rows; r++) {
            fs.write(mat.ptr<char>(r), mat.cols * mat.elemSize());
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


std::optional<Frame> FileLoader::readFrame(const std::string &videoName, size_t index) const
{
    auto def = cv::Mat();
    ifstream stream(rootDir / videoName / "frames" / (to_string(index) + ".keypoints"), fstream::binary);

    auto features = readFrame(videoName, index, Features);
    auto bag = readFrame(videoName, index, Descriptor);
    auto color = readFrame(videoName, index, ColorHistogram);
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

std::optional<cv::Mat> FileLoader::readFrame(const std::string &videoName, size_t index, FrameDataType type) const
{
    switch(type) {
        case ColorHistogram: return readFrameData(videoName, to_string(index) + ".color");
        case Descriptor: return readFrameData(videoName, to_string(index) + ".bag");
        case Features: return readFrameData(videoName, to_string(index) + ".sift");
    }

    return std::nullopt;
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

    return saveFrame(video, index, Features, frame.descriptors) &&
           saveFrame(video, index, ColorHistogram, frame.colorHistogram) &&
           saveFrame(video, index, Descriptor, frame.frameDescriptor);
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

bool FileLoader::saveFrame(const std::string &videoName, size_t index, FrameDataType type, const cv::Mat &mat) const
{
    switch(type) {
        case ColorHistogram: return saveFrameData(videoName, to_string(index) + ".color", mat);
        case Descriptor: return saveFrameData(videoName, to_string(index) + ".bag", mat);
        case Features: return saveFrameData(videoName, to_string(index) + ".sift", mat);
    }

    return false;
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

size_t FileLoader::countFrames(const std::string& video) const {
    size_t count = 0;
    for(auto& entry : fs::directory_iterator(rootDir / video / "frames")) {
        if(entry.path().extension() == ".sift")
            count++;
    }

    return count;
}
size_t FileLoader::countScenes(const std::string& video) const {
    size_t count = 0;
    for(auto& entry : fs::directory_iterator(rootDir / video / "scenes")) {
        count++;
    }

    return count;
}

void clearDir(fs::path path)
{
    if (fs::exists(path))
    {
        fs::remove_all(path);
    }

    fs::create_directories(path);
}

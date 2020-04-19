#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <raft>
#include <utility>
#include <opencv2/core/mat.hpp>
#include "vocab_type.hpp"
#include "video.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include "storage.hpp"
#include <memory>
#include <string>
#include <queue>
#include <type_traits>

typedef v_size size_type;
typedef std::pair<size_type, size_type> scene_range;
typedef ordered_adapter<SerializableScene, v_size> ordered_scene;
typedef ordered_adapter<cv::Mat, v_size> ordered_mat;
typedef ordered_adapter<Frame, v_size> ordered_frame;

template <typename T>
class Duplicate : public raft::kernel
{
public:
    Duplicate() : raft::kernel()
    {
        input.addPort<T>("in");
        output.addPort<T>("first");
        output.addPort<T>("second");
    }

    raft::kstatus run()
    {
        auto thing = input["in"].peek<T>();

        output["first"].allocate<T>(thing);
        output["second"].allocate<T>(thing);
        output["first"].send();
        output["second"].send();

        input["in"].unpeek();
        input["in"].recycle();
        return raft::proceed;
    }
};

template <typename T>
class Null : public raft::kernel
{
public:
    Null() : raft::kernel()
    {
        input.addPort<T>("in");
    }

    raft::kstatus run()
    {
        input["in"].recycle();
        return raft::proceed;
    }
};

class VideoFrameSource : public raft::kernel
{
    cv::VideoCapture cap;
    size_type counter = 0;

public:
    VideoFrameSource(const std::string &path) : raft::kernel(), cap(path, cv::CAP_ANY)
    {
        output.addPort<ordered_mat>("image");
    }

    raft::kstatus run();
};

class ScaleImage : public raft::kernel
{
    std::pair<std::uint16_t, std::uint16_t> cropsize;

public:
    ScaleImage(std::pair<std::uint16_t, std::uint16_t> cropsize) : raft::kernel(), cropsize(cropsize)
    {
        input.addPort<ordered_mat>("image");
        output.addPort<ordered_mat>("scaled_image");
    }

    raft::kstatus run();
};

class ExtractSIFT : public raft::kernel
{
    cv::Ptr<cv::FeatureDetector> detector;

public:
    ExtractSIFT();

    raft::kstatus run();
};

class ExtractColorHistogram : public raft::kernel
{
public:
    ExtractColorHistogram() : raft::kernel()
    {
        input.addPort<ordered_mat>("image");
        output.addPort<ordered_mat>("color_histogram");
    }

    raft::kstatus run();
};

class DetectScene : public raft::kernel
{
    size_type windowSize;
    size_type lastMarker = 0;
    size_type currentScene = 0;
    std::queue<double> responses;
    double previousAverage = 0;
    double accumulatedResponse = 0;
    double threshold;
    ordered_mat prev;

public:
    DetectScene(double threshold, size_type windowSize = 10) : raft::kernel(), windowSize(windowSize),
                                                               threshold(threshold)
    {
        input.addPort<ordered_mat>("color_histogram");
        output.addPort<scene_range>("scene_range");
    }

    raft::kstatus run();
};

class ExtractFrame : public raft::kernel
{
    Vocab<Frame> frameVocab;

public:
    ExtractFrame(const Vocab<Frame> &frameVocab) : raft::kernel(), frameVocab(frameVocab)
    {
        input.addPort<ordered_mat>("sift_descriptor");
        output.addPort<ordered_mat>("frame_descriptor");
    }

    raft::kstatus run();
};

class ExtractScene : public raft::kernel
{
    Vocab<SerializableScene> sceneVocab;
    size_type counter = 0;

public:
    ExtractScene(const Vocab<SerializableScene> &sceneVocab) : raft::kernel(), sceneVocab(sceneVocab)
    {
        input.addPort<ordered_mat>("frame_descriptor");
        input.addPort<scene_range>("scene_range");
        output.addPort<ordered_scene>("scene");
    };

    raft::kstatus run();
};

class SaveSIFT : public raft::kernel
{
    FileLoader loader;
    std::string video;

public:
    SaveSIFT(const std::string &video, const FileLoader &loader) : raft::kernel(),
                                                                   loader(loader), video(video)
    {
        input.addPort<ordered_mat>("sift_descriptor");

        loader.initVideoDir(video);
    }

    raft::kstatus run();
};

class SaveColor : public raft::kernel
{
    FileLoader loader;
    std::string video;

public:
    SaveColor(const std::string &video, const FileLoader &loader) : raft::kernel(),
                                                                    loader(loader), video(video)
    {
        input.addPort<ordered_mat>("color_histogram");

        loader.initVideoDir(video);
    }

    raft::kstatus run();
};

class SaveFrameDescriptor : public raft::kernel
{
    FileLoader loader;
    std::string video;

public:
    SaveFrameDescriptor(const std::string &video, const FileLoader &loader) : raft::kernel(),
                                                                              loader(loader), video(video)
    {
        input.addPort<ordered_mat>("frame_descriptor");

        loader.initVideoDir(video);
    }

    raft::kstatus run();
};

class SaveScene : public raft::kernel
{
    FileLoader loader;
    std::string video;

public:
    SaveScene(const std::string &video, const FileLoader &loader) : raft::kernel(),
                                                                    loader(loader), video(video)
    {
        input.addPort<ordered_scene>("scene");

        loader.initVideoDir(video);
    }

    raft::kstatus run();
};

class LoadScenes : public raft::kernel
{
    FileLoader loader;
    std::string video;

public:
    LoadScenes(const std::string &video, const FileLoader &loader) : raft::kernel(),
                                                                     loader(loader), video(video)
    {
        output.addPort<ordered_scene>("scene");

        loader.initVideoDir(video);
    }

    raft::kstatus run();
};

class DebugDisplay : public raft::kernel
{
public:
    DebugDisplay();

    raft::kstatus run();
};

#endif
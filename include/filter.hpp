#ifndef KERNEL_HPP
#define KERNEL_HPP

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
#include <tbb/pipeline.h>

typedef v_size size_type;
typedef std::pair<size_type, size_type> scene_range;
typedef ordered_adapter<SerializableScene, v_size> ordered_scene;
typedef ordered_adapter<cv::Mat, v_size> ordered_mat;
typedef ordered_adapter<Frame, v_size> ordered_frame;

class VideoFrameSource
{
    cv::VideoCapture cap;
    size_type counter = 0;

public:
    size_t totalFrames;

    VideoFrameSource(const std::string &path) : cap(path, cv::CAP_ANY)
    {
        totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    }

    ordered_mat operator()(tbb::flow_control &);
};

class ScaleImage
{
    std::pair<std::uint16_t, std::uint16_t> cropsize;

public:
    ScaleImage(std::pair<std::uint16_t, std::uint16_t> cropsize) : cropsize(cropsize) {}

    ordered_mat& operator()(ordered_mat&) const;
};

class ExtractSIFT
{
    cv::Ptr<cv::FeatureDetector> detector;

public:
    ExtractSIFT();

    ordered_mat operator()(const ordered_mat&) const;
};

class ExtractColorHistogram
{
public:
    ExtractColorHistogram() {}

    ordered_mat operator()(const ordered_mat&) const;
};

class ExtractFrame
{
    Vocab<Frame> frameVocab;

public:
    ExtractFrame(const Vocab<Frame> &frameVocab) : frameVocab(frameVocab) {}

    ordered_mat operator()(const ordered_mat&) const;
};

class SaveFrameSink
{
    FileLoader loader;
    std::string video;

public:
    SaveFrameSink(const std::string &video, const FileLoader &loader) : loader(loader),
                                                                        video(video)
    {
        loader.initVideoDir(video);
    }

    void operator()(const ordered_frame&) const;
};

#endif
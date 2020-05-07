#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <utility>
#include <opencv2/core/mat.hpp>
#include "vocab_type.hpp"
#include "video.hpp"
#include <opencv2/features2d.hpp>
#include "storage.hpp"
#include <memory>
#include <string>

typedef size_t size_type;
typedef std::pair<size_type, size_type> scene_range;
typedef ordered_adapter<SerializableScene, size_t> ordered_scene;
typedef ordered_adapter<cv::Mat, size_t> ordered_mat;
typedef ordered_adapter<cv::UMat, size_t> ordered_umat;
typedef ordered_adapter<Frame, size_t> ordered_frame;

class ScaleImage
{
    std::pair<std::uint16_t, std::uint16_t> cropsize;

public:
    ScaleImage(std::pair<std::uint16_t, std::uint16_t> cropsize) : cropsize(cropsize) {}

    ordered_umat& operator()(ordered_umat&) const;
    cv::UMat& operator()(cv::UMat&) const;
};

class ExtractSIFT
{
    cv::Ptr<cv::FeatureDetector> detector;

public:
    ExtractSIFT();

    ordered_mat operator()(const ordered_umat&) const;
    cv::Mat operator()(const cv::UMat&) const;
    Frame withKeyPoints(const cv::UMat&) const;
};

class ExtractColorHistogram
{
public:
    ExtractColorHistogram() {}

    ordered_mat operator()(const ordered_umat&) const;
    cv::Mat operator()(const cv::UMat&) const;
};

class ExtractFrame
{
    Vocab<Frame> vocab;

public:
    ExtractFrame(const Vocab<Frame> &frameVocab);

    ordered_mat operator()(const ordered_mat&) const;
    cv::Mat operator()(const cv::Mat&) const;
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
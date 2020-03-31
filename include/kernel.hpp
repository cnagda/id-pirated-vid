#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <raft>
#include <utility>
#include <opencv2/core/mat.hpp>
#include "vocab_type.hpp"
#include "frame.hpp"
#include "scene.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <memory>

typedef size_t size_type;
typedef scene_range scene_range;

class VideoFrameSource : public raft::kernel {
    cv::VideoCapture cap;
public: 
    VideoFrameSource(const std::string& path) : raft::kernel(), cap(path, cv::CAP_ANY) {
        output.addPort<cv::Mat>("image");
    }

    raft::kstatus run();
};

class ScaleImage : public raft::kernel {
    std::pair<std::uint16_t, std::uint16_t> cropsize;
public:
    ScaleImage(std::pair<std::uint16_t, std::uint16_t> cropsize): raft::kernel(), cropsize(cropsize) {
        input.addPort<cv::Mat>("image");
        output.addPort<cv::Mat>("scaled_image");
    }

    raft::kstatus run();
};

class ExtractSIFT : public raft::kernel {
    std::unique_ptr<cv::FeatureDetector> detector;
public:
    ExtractSIFT();

    raft::kstatus run();
};

class ExtractColorHistogram : public raft::kernel {
public:
    ExtractColorHistogram(): raft::kernel() {
        input.addPort<cv::Mat>("image");
        output.addPort<cv::Mat>("color_histogram");
    }

    raft::kstatus run();
};

class DetectScene : public raft::kernel {
public:
    DetectScene(): raft::kernel() {
        input.addPort<cv::Mat>("color_histogram");
        output.addPort<scene_range>("scene_range");
    }

    raft::kstatus run();
};

class ExtractFrame : public raft::kernel {
    Vocab<Frame> frameVocab;
public:
    ExtractFrame(const Vocab<Frame>& frameVocab): raft::kernel(), frameVocab(frameVocab) {
        input.addPort<cv::Mat>("sift_descriptor");
        output.addPort<cv::Mat>("frame_descriptor");
    }

    raft::kstatus run();
};

class ExtractScene : public raft::kernel {
    Vocab<SerializableScene> sceneVocab;
public:
    ExtractScene(const Vocab<SerializableScene>& sceneVocab): raft::kernel(), sceneVocab(sceneVocab) {
        input.addPort<cv::Mat>("sift_descriptor");
        input.addPort<scene_range>("scene_range");
        output.addPort<SerializableScene>("scene");
    };

    raft::kstatus run();
};

class CollectFrame : public raft::kernel {
public:
    CollectFrame(): raft::kernel() {
        
    }
};

class DebugDisplay : public raft::kernel {
public:
    DebugDisplay();

    raft::kstatus run();
};

#endif
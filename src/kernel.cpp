#include "kernel.hpp"
#include <raft>
#include "imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "vocabulary.hpp"
#include "matcher.hpp"

#define HBINS 32
#define SBINS 30

raft::kstatus VideoFrameSource::run() {
    cv::Mat image;

    if(cap.read(image)) {
        output["image"].push(image);
        return raft::proceed;
    }

    return raft::stop;
}

raft::kstatus ScaleImage::run() {
    auto image = output["scaled_image"].allocate<cv::Mat>();
    input["image"].pop(image);
    image = scaleToTarget(image, cropsize.first, cropsize.second);
    output["scaled_image"].send();

    return raft::proceed;
}

ExtractSIFT::ExtractSIFT() : raft::kernel(), detector(cv::xfeatures2d::SiftFeatureDetector::create(500)) {
    input.addPort<cv::Mat>("image");
    output.addPort<cv::Mat>("sift_descriptor");
    output.addPort<size_type>("keypoints_size");
    output.addPort<cv::KeyPoint>("keypoints");
}

raft::kstatus ExtractSIFT::run() {
    auto image = input["image"].peek<cv::Mat>();
    std::vector<cv::KeyPoint> keyPoints;
    auto descriptors = output["sift_descriptor"].allocate<cv::Mat>();
    
    detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);
    input["image"].unpeek();
    input["image"].recycle();

    output["sift_descriptor"].send();
    output["keypoints_size"].push(keyPoints.size());
    output["keypoints"].insert(keyPoints.begin(), keyPoints.end());

    return raft::proceed;
}

raft::kstatus ExtractColorHistogram::run() {
    auto image = input["image"].peek<cv::Mat>();
    cv::UMat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    input["image"].unpeek();
    input["image"].recycle();

    auto colorHistogram = output["color_histogram"].allocate<cv::Mat>();

    std::vector<int> histSize{HBINS, SBINS};
    // hue varies from 0 to 179, see cvtColor
    std::vector<float> ranges{ 0, 180, 0, 256 };
    // we compute the histogram from the 0-th and 1-st channels
    std::vector<int> channels{0, 1};

    cv::calcHist( std::vector<decltype(hsv)>{hsv}, channels, cv::Mat(), // do not use mask
                    colorHistogram, histSize, ranges,
                    true);
    cv::normalize( colorHistogram, colorHistogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    output["color_histogram"].send();

    return raft::proceed;
}

raft::kstatus ExtractFrame::run() {
    auto sift = input["sift_descriptor"].peek<cv::Mat>();
    auto descriptor = baggify(sift, frameVocab.descriptors());
    input["sift_descriptor"].unpeek();
    input["sift_descriptor"].recycle();

    output["frame_descriptor"].push(descriptor);
    return raft::proceed;
}

raft::kstatus ExtractScene::run() {
    scene_range range;
    input["scene_range"].pop(range);

    auto length = range.second - range.first;
    decltype(length) index = 0;

    std::vector<cv::Mat> descriptors(length);

    while(index != length) input["frame_descriptor"].pop(descriptors[index++]);

    auto descriptor = baggify(descriptors, sceneVocab.descriptors());

    output["scene"].allocate<SerializableScene>(descriptor, range.first, range.second);
    output["scene"].send();

    return raft::proceed;
}

raft::kstatus CollectFrame::run() {
    cv::Mat siftDescriptor, frameDescriptor;

    input["sift_descriptor"].pop(siftDescriptor);
    input["frame_descriptor"].pop(frameDescriptor);

    output["frame"].allocate<Frame>(std::vector<cv::KeyPoint>{}, siftDescriptor, frameDescriptor, cv::Mat());
    output["frame"].send();

    return raft::proceed;
}

raft::kstatus SaveFrame::run() {
    auto frame = input["frame"].peek<ordered_frame>();

    loader.saveFrame(video, frame.rank, frame.data);

    input["frame"].unpeek();
    input["frame"].recycle();

    return raft::proceed;
}

raft::kstatus SaveScene::run() {
    auto scene = input["scene"].peek<ordered_scene>();

    loader.saveScene(video, scene.rank, scene.data);

    input["scene"].unpeek();
    input["scene"].recycle();

    return raft::proceed;
}

raft::kstatus DetectScene::run() {
    ColorComparator comp;
    cv::Mat mat;

    if(responses.size() != windowSize) {
        input["color_histogram"].pop(prev);

        while(responses.size() != windowSize) {
            input["color_histogram"].pop(mat);
            double response = comp(mat, prev);

            accumulatedResponse += response;
            responses.push(response);
            prev = mat;
            currentScene++;
        }

        previousAverage = accumulatedResponse / windowSize;
    }

    input["color_histogram"].pop(mat);
    double response = comp(mat, prev);
    accumulatedResponse += response;
    accumulatedResponse -= responses.front();

    if(std::abs(accumulatedResponse / windowSize - previousAverage) > threshold) {
        auto range = output["scene_range"].allocate<scene_range>(lastMarker, currentScene);
        output["scene_range"].send();
        lastMarker = currentScene;
    }

    previousAverage = accumulatedResponse / windowSize;

    responses.push(response);
    responses.pop();
    prev = mat;
    currentScene++;
    
    return raft::proceed;
}
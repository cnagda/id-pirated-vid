#include "kernel.hpp"
#include <raft>
#include "imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "vocabulary.hpp"
#include "matcher.hpp"
#include <cassert>

#define HBINS 32
#define SBINS 30

raft::kstatus VideoFrameSource::run() {
    ordered_mat image;

    if(cap.read(image.data)) {
        image.rank = counter++;
        output["image"].push(image);
        return raft::proceed;
    }

    std::cout << "stopping" << std::endl;
    return raft::stop;
}

raft::kstatus ScaleImage::run() {
    auto image = input["image"].peek<ordered_mat>();
    image.data = scaleToTarget(image.data, cropsize.first, cropsize.second);
    output["scaled_image"].push(image);

    input["image"].unpeek();
    input["image"].recycle();
    return raft::proceed;
}

ExtractSIFT::ExtractSIFT() : raft::kernel(), detector(cv::xfeatures2d::SiftFeatureDetector::create(500)) {
    input.addPort<ordered_mat>("image");
    output.addPort<ordered_mat>("sift_descriptor");
    // output.addPort<size_type>("keypoints_size");
    // output.addPort<cv::KeyPoint>("keypoints");
}

raft::kstatus ExtractSIFT::run() {
    auto image = input["image"].peek<ordered_mat>();
    std::vector<cv::KeyPoint> keyPoints;
    ordered_mat descriptors;
    descriptors.rank = image.rank;

    detector->detectAndCompute(image.data, cv::noArray(), keyPoints, descriptors.data);
    input["image"].unpeek();
    input["image"].recycle();

    output["sift_descriptor"].push(descriptors);
    //output["keypoints_size"].push(keyPoints.size());
    //output["keypoints"].insert(keyPoints.begin(), keyPoints.end());

    return raft::proceed;
}

raft::kstatus ExtractColorHistogram::run() {
    auto image = input["image"].peek<ordered_mat>();
    cv::UMat hsv;
    cv::cvtColor(image.data, hsv, cv::COLOR_BGR2HSV);

    input["image"].unpeek();
    input["image"].recycle();

    ordered_mat colorHistogram;
    colorHistogram.rank = image.rank;

    std::vector<int> histSize{HBINS, SBINS};
    // hue varies from 0 to 179, see cvtColor
    std::vector<float> ranges{ 0, 180, 0, 256 };
    // we compute the histogram from the 0-th and 1-st channels
    std::vector<int> channels{0, 1};

    cv::calcHist( std::vector<decltype(hsv)>{hsv}, channels, cv::Mat(), // do not use mask
                    colorHistogram.data, histSize, ranges,
                    true);
    cv::normalize( colorHistogram.data, colorHistogram.data, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

    output["color_histogram"].push(colorHistogram);

    return raft::proceed;
}

raft::kstatus ExtractFrame::run() {
    auto sift = input["sift_descriptor"].peek<ordered_mat>();
    auto rank = sift.rank;
    auto descriptor = baggify(sift.data, frameVocab.descriptors());
    input["sift_descriptor"].unpeek();
    input["sift_descriptor"].recycle();

    output["frame_descriptor"].allocate<ordered_mat>(rank, descriptor);
    output["frame_descriptor"].send();
    return raft::proceed;
}

raft::kstatus ExtractScene::run() {
    scene_range range;
    input["scene_range"].pop(range);

    auto length = range.second - range.first;
    std::cout << "length " << length << std::endl;
    decltype(length) index = 0;

    std::vector<cv::Mat> descriptors(length);

    while(index != length) {
        auto frame = input["frame_descriptor"].peek<ordered_mat>();
        std::cout << "index " << index << std::endl;
        descriptors[index++] = frame.data;
        input["frame_descriptor"].unpeek();
        input["frame_descriptor"].recycle();
    }

    auto descriptor = baggify(descriptors.begin(), descriptors.end(), sceneVocab.descriptors());

    output["scene"].allocate<ordered_scene>(counter++, SerializableScene{descriptor, range.first, range.second});
    output["scene"].send();

    return raft::proceed;
}

raft::kstatus CollectFrame::run() {
    ordered_mat siftDescriptor, frameDescriptor;

    input["sift_descriptor"].pop(siftDescriptor);
    input["frame_descriptor"].pop(frameDescriptor);
    assert(siftDescriptor.rank == frameDescriptor.rank);

    output["frame"].allocate<ordered_frame>(frameDescriptor.rank, Frame{siftDescriptor.data, frameDescriptor.data, cv::Mat()});
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
    std::cout << "scene rank " << scene.rank << std::endl;

    loader.saveScene(video, scene.rank, scene.data);

    input["scene"].unpeek();
    input["scene"].recycle();

    return raft::proceed;
}

raft::kstatus DetectScene::run() {
    ColorComparator comp;
    ordered_mat mat;

    if(responses.size() != windowSize) {
        input["color_histogram"].pop(prev);

        while(responses.size() != windowSize) {
            input["color_histogram"].pop(mat);
            double response = comp(mat.data, prev.data);

            accumulatedResponse += response;
            responses.push(response);
            prev = mat;
            currentScene++;
        }

        previousAverage = accumulatedResponse / windowSize;
    }

    input["color_histogram"].pop(mat);
    double response = comp(mat.data, prev.data);
    accumulatedResponse += response;
    accumulatedResponse -= responses.front();

    if(std::abs(accumulatedResponse / windowSize - previousAverage) > threshold) {
        output["scene_range"].allocate<scene_range>(lastMarker, currentScene);
        output["scene_range"].send();
        lastMarker = currentScene;
    }

    previousAverage = accumulatedResponse / windowSize;

    responses.push(response);
    responses.pop();
    prev = mat;

    std::cout << "scene " << currentScene << std::endl;
    currentScene++;

    if(input["color_histogram"].is_invalid() && input["color_histogram"].size() == 0) {
        std::cout << "done" << std::endl;
        output["scene_range"].allocate<scene_range>(lastMarker, currentScene);
        output["scene_range"].send();
        lastMarker = currentScene;
    }
    
    return raft::proceed;
}
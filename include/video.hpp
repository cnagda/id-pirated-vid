#ifndef VIDEO_HPP
#define VIDEO_HPP
#include <vector>
#include <string>
#include <functional>
#include "frame.hpp"

struct SIFTVideo {
    using size_type = std::vector<Frame>::size_type;

    std::vector<Frame> SIFTFrames;
    SIFTVideo() : SIFTFrames() {};
    SIFTVideo(const std::vector<Frame>& frames) : SIFTFrames(frames) {};
    SIFTVideo(std::vector<Frame>&& frames) : SIFTFrames(frames) {};
    std::vector<Frame>& frames() & { return SIFTFrames; };
    size_type frameCount() { return SIFTFrames.size(); };
};

SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr, std::pair<int, int> cropsize = {600, 700});

#endif
#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <functional>

void SIFTwrite(const std::string& filename, const cv::Mat& mat, const std::vector<cv::KeyPoint>& keyPoints);
std::pair<cv::Mat, std::vector<cv::KeyPoint>> SIFTread(const std::string& filename);
std::string getAlphas(std::string input);
void createFolder(std::string folder_name);

class IVideo {
public:
    virtual std::vector<Frame> frames() = 0;
};

class SIFTVideo : public IVideo {
private:
    std::vector<Frame> SIFTFrames;
public:
    SIFTVideo(const std::vector<Frame>& frames) : SIFTFrames(frames) {};
    SIFTVideo(std::vector<Frame>&& frames) : SIFTFrames(frames) {};
    std::vector<Frame> frames() { return SIFTFrames; };
};

class IDatabase {
public:
    virtual std::unique_ptr<IVideo> addVideo(const std::string& filepath, std::function<void(cv::Mat, Frame)> callback = nullptr) = 0;
    virtual std::unique_ptr<IVideo> loadVideo(const std::string& filepath) = 0;
};

class FileDatabase : public IDatabase {
public:
    std::unique_ptr<IVideo> addVideo(const std::string& filepath, std::function<void(cv::Mat, Frame)> callback = nullptr);
    std::unique_ptr<IVideo> loadVideo(const std::string& filepath);
};

#endif
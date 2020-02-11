#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <functional>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(std::string input);
void createFolder(std::string folder_name);

class IVideo {
public:
    virtual std::vector<Frame> frames() = 0;
    virtual ~IVideo() = default;
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
    virtual std::unique_ptr<IVideo> loadVideo(const std::string& filepath) const = 0;
    virtual std::vector<std::string> listVideos() const = 0;
    virtual ~IDatabase() = default;
};

class FileDatabase : public IDatabase {
private:
    fs::path databaseRoot;
public:
    FileDatabase() : FileDatabase(fs::current_path() / "database") {};
    FileDatabase(const std::string& databasePath);
    std::unique_ptr<IVideo> addVideo(const std::string& filepath, std::function<void(cv::Mat, Frame)> callback = nullptr) override;
    std::unique_ptr<IVideo> loadVideo(const std::string& filepath) const override;
    std::vector<std::string> listVideos() const override;
};

#endif
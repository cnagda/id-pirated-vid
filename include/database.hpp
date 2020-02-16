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

class SIFTVideo;

void SIFTwrite(const std::string& filename, const Frame& frame);
Frame SIFTread(const std::string& filename);
std::string getAlphas(const std::string& input);
void createFolder(const std::string& folder_name);
SIFTVideo getSIFTVideo(const std::string& filename, std::function<void(cv::Mat, Frame)> callback = nullptr);

class IVideo {
public:
    IVideo(const std::string& name) : name(name) {};
    const std::string name;
    using size_type = std::vector<Frame>::size_type;

    virtual size_type frameCount() = 0;
    virtual std::vector<Frame>& frames() = 0;
    virtual ~IVideo() = default;
};

class SIFTVideo : public IVideo {
private:
    std::vector<Frame> SIFTFrames;
public:
    SIFTVideo(const std::string& name, const std::vector<Frame>& frames) : IVideo(name), SIFTFrames(frames) {};
    SIFTVideo(const std::string& name, std::vector<Frame>&& frames) : IVideo(name), SIFTFrames(frames) {};
    SIFTVideo(SIFTVideo&& vid) : IVideo(vid.name), SIFTFrames(vid.SIFTFrames) {};
    std::vector<Frame>& frames() override { return SIFTFrames; };
    size_type frameCount() override { return SIFTFrames.size(); };
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
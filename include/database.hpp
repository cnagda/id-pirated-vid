#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"
#include <vector>
#include <memory>
#include <string>

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
    virtual std::unique_ptr<IVideo> addVideo(const std::string& filepath) = 0;
    virtual std::unique_ptr<IVideo> loadVideo(const std::string& filepath) = 0;
};

class FileDatabase : public IDatabase {
public:
    std::unique_ptr<IVideo> addVideo(const std::string& filepath);
    std::unique_ptr<IVideo> loadVideo(const std::string& filepath);
};

#endif
#ifndef DATABASE_H
#define DATABASE_H
#include "frame.hpp"

class IVideo {
public:
    Frame seekFrame();
};

class IDatabase {
public:
    std::unique_ptr<IVideo> addVideo();
    std::unique_ptr<IVideo> loadVideo();
};

#endif
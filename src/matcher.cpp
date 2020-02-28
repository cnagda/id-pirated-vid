#include <fstream>
#include <iostream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"
#include "kmeans2.hpp"
#include "sw.hpp"
#include "vocabulary.hpp"

Vocab<Frame> constructFrameVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator) {
    cv::Mat descriptors;

    for(auto &video : database.loadVideo()) {
        auto frames = video->frames();
        for(auto i = frames.begin(); i != frames.end(); i += speedinator)
                descriptors.push_back(i->descriptors);
    }

    return Vocab<Frame>(constructVocabulary(descriptors, K));
}

Vocab<IScene> constructSceneVocabulary(const FileDatabase& database, unsigned int K, unsigned int speedinator) {
    auto vocab = loadVocabulary<Vocab<Frame>>(database);
    if(!vocab) {
        throw std::runtime_error("trying to construct frame vocab but sift vocab is empty");
    }
    auto d = vocab->descriptors();

    cv::Mat descriptors;

    for(auto &video : database.loadVideo()) {
        auto frames = video->frames();
        for(auto i = frames.begin(); i != frames.end(); i += speedinator)
                descriptors.push_back(baggify(i->descriptors, d));
    }

    return Vocab<IScene>(constructVocabulary(descriptors, K));
}
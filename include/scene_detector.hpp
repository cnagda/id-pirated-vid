#ifndef SCENE_DETECT_HPP
#define SCENE_DETECT_HPP

#include <vector>
#include <numeric>

template<class Video, typename Cmp>
auto convolutionalDetector(Video& video, Cmp&& comp, double threshold, int windowSize = 10){
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto& frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    std::vector<double> responses;
    std::vector<double> conv;

    for(int i = 1; i < frames.size(); i++) {
        responses.push_back(comp(frames[i], frames[i - 1]));
    }

    double sum = 0;
    for(int i = 0; i < windowSize; i++) sum += responses[i];
    conv.push_back(sum / windowSize);

    for(int i = windowSize; i < responses.size(); i++) {
        sum += responses[i];
        sum -= responses[i - windowSize];

        conv.push_back(sum / windowSize);
    }

    index_t last = 0;

    for(int i = 1; i < conv.size(); ++i) {
        auto dif = std::abs(conv[i] - conv[i - 1]);
        if(dif > threshold) {
            auto end = i + windowSize;
            retval.push_back({last, end});
            last = end;
        }
    }

    retval.push_back({last, frames.size()});
    
    return retval;
}

#endif
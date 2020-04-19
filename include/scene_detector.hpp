#ifndef SCENE_DETECT_HPP
#define SCENE_DETECT_HPP

#include <vector>
#include <utility>

template<class Video, typename Cmp>
auto flatScenes(Video& video, Cmp&& comp, double threshold){
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto& frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    index_t last = 0;

    // for(int i = 1; i < frames.size(); i++) {
    //     if(double fs = comp(frames[i], frames[i - 1]); fs < threshold){
    //         if(!frames[i].descriptors.empty()){ // do not include black frames
    //             retval.push_back({last, i});
    //             last = i;
    //         }
    //     } else {
    //         // std::cout << "i: " << i << "sim: " << fs << std::endl;
    //     }
    // }

    for(int i = FRAMES_PER_SCENE - 1; i < frames.size() - FRAMES_PER_SCENE; i+=FRAMES_PER_SCENE) {
        while (frames[i].descriptors.empty() && i < frames.size() - FRAMES_PER_SCENE) { i++; }
        if (i == frames.size()) { break; }
        retval.push_back({last, i});
        last = i;
    }
    if(!frames.back().descriptors.empty()){ // do not include black frames
        retval.push_back({last, frames.size()});
    }

    return retval;
}

std::vector<float> get_distances(FileLoader fl, std::string video_path);
std::vector<std::pair<int, int>> hierarchicalScenes(std::vector<float> distances, int min_scene_length);

/*std::vector<float> get_distances(FileLoader fl, std::string video_path){
    
    cv::Mat current, next;
    std::vector<float> retval;

    auto rv = fl.readFrameColorHistogram(video_path, 0);
    if(rv){
        current = *rv;
    }
    else{
        return {}; 
    }

    int index = 0;
    ColorComparator cc;

    while(1){
        index++;
        rv = fl.readFrameColorHistogram(video_path, index);
        if(rv){
            next = *rv;
            retval.push_back(cc(current, next));
            current = next;
        }
        else{
            return retval;
        }
    }
}

std::vector<int> hierarchicalScenes(std::vector<float> distances, int min_scene_length){
    std::vector<bool> excluded(distances.size(), 0);
    std::vector<std::pair<float, int>> sorted_distances;

    for(int i = 0; i < distances.size(); i++){
        sorted_distances.push_back({distances[i], i});
    }
    
    std::vector<int> retval;
    std::sort(sorted_distances.begin(), sorted_distances.end(), [](auto l, auto r) { return l.first > r.first; } );

    int index = -1;
    std::pair<float, int> current;

    // iterate over distances from lagest to smallest
    while(index < distances.size()){
        index++;
        current = sorted_distances[index];
        if(excluded[index]){
            continue;
        }

        retval.push_back(current.second);
        int low = std::max(index - min_scene_length, 0);
        int high = std::min(index + min_scene_length, (int)distances.size());

        // exclude neighbors of current
        for(int i = low; i < high; i++){
            excluded[i] = 1;
        }

    }

    // return cutoffs from left to right
    std::sort(retval.begin(), retval.end());

    return {};
}*/

template<class Video, typename Cmp>
auto convolutionalDetector(Video& video, Cmp&& comp, double threshold, unsigned int windowSize = 10){
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto& frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    std::vector<double> responses;
    std::vector<double> conv;

    for(unsigned int i = 1; i < frames.size(); i++) {
        responses.push_back(comp(frames[i], frames[i - 1]));
    }

    double sum = 0;
    for(unsigned int i = 0; i < windowSize; i++) sum += responses[i];
    conv.push_back(sum / windowSize);

    for(unsigned int i = windowSize; i < responses.size(); i++) {
        sum += responses[i];
        sum -= responses[i - windowSize];

        conv.push_back(sum / windowSize);
    }

    index_t last = 0;

    for(unsigned int i = 1; i < conv.size(); ++i) {
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

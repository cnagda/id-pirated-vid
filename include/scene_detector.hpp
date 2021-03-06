#ifndef SCENE_DETECT_HPP
#define SCENE_DETECT_HPP

#include <vector>
#include <utility>

#define FRAMES_PER_SCENE 45

// FIXME broken

/*
template <class Video, typename Cmp>
auto flatScenes(Video &video, Cmp &&comp, double threshold)
{
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto frames = video.frames();

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

    for (int i = FRAMES_PER_SCENE - 1; i < frames.size() - FRAMES_PER_SCENE; i += FRAMES_PER_SCENE)
    {
        while (frames[i].descriptors.empty() && i < frames.size() - FRAMES_PER_SCENE)
        {
            i++;
        }
        if (i == frames.size())
        {
            break;
        }
        retval.push_back({last, i});
        last = i;
    }
    if (!frames.back().descriptors.empty())
    { // do not include black frames
        retval.push_back({last, frames.size()});
    }

    return retval;
} */

template<typename Cmp, typename Source>
std::vector<double> get_distances(Source&& fl, Cmp&& comp)
{
    std::vector<double> retval;
    auto prev = fl.read();

    while (auto current = fl.read())
    {
        retval.push_back(comp(*current, *prev));
        prev = current;
    }

    return retval;
}

std::vector<std::pair<unsigned int, unsigned int>> hierarchicalScenes(const std::vector<double>& distances, int min_scene_length);
std::vector<std::pair<unsigned int, unsigned int>> thresholdScenes(const std::vector<double>& distances, double threshold);

template <class Video, typename Cmp>
auto convolutionalDetector(Video &video, Cmp &&comp, double threshold, unsigned int windowSize = 10)
{
    typedef typename std::decay_t<Video>::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<index_t, index_t>> retval;

    auto &frames = video.frames();
    if (!frames.size())
    {
        return retval;
    }

    std::vector<double> responses;
    std::vector<double> conv;

    for (unsigned int i = 1; i < frames.size(); i++)
    {
        responses.push_back(comp(frames[i], frames[i - 1]));
    }

    double sum = 0;
    for (unsigned int i = 0; i < windowSize; i++)
        sum += responses[i];
    conv.push_back(sum / windowSize);

    for (unsigned int i = windowSize; i < responses.size(); i++)
    {
        sum += responses[i];
        sum -= responses[i - windowSize];

        conv.push_back(sum / windowSize);
    }

    index_t last = 0;

    for (unsigned int i = 1; i < conv.size(); ++i)
    {
        auto dif = std::abs(conv[i] - conv[i - 1]);
        if (dif > threshold)
        {
            auto end = i + windowSize;
            retval.push_back({last, end});
            last = end;
        }
    }

    retval.push_back({last, frames.size()});

    return retval;
}

#endif

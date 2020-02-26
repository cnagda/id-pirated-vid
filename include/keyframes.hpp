#ifndef KEYFRAMES_HPP
#define KEYFRAMES_HPP

#include "concepts.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "instrumentation.hpp"

template <class T, class RankType>
struct sortable{
    RankType rank;
    T data;
    bool operator<(const sortable& a) const {  return rank < a.rank; }; 
};

template<typename Cmp>
std::vector<std::pair<IVideo::size_type, IVideo::size_type>> flatScenes(IVideo& video, Cmp comp, double threshold){
    typedef IVideo::size_type index_t;
    std::cout << "In flatScenes" << std::endl;

    std::vector<std::pair<IVideo::size_type, IVideo::size_type>> retval;    

    auto& frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    index_t last = 0;

    for(int i = 1; i < frames.size(); i++) {
        if(double fs = comp(frames[i], frames[i - 1]) < threshold){
            if(!frames[i].descriptors.empty()){ // do not include black frames
                retval.push_back({last, i - 1});
                last = i;
            }
        }
    }
    
    return retval;
}

template<typename RangeIt,
    typename = std::enable_if_t<is_pair_iterator_v<RangeIt>>>
std::vector<cv::Mat> flatScenesBags(RangeIt start, RangeIt end, cv::Mat frameVocab) {
    static_assert(is_pair_iterator_v<RangeIt>, 
        "flatScenesBags requires an iterator to a pair");

    std::cout << "In flatScenesBags" << std::endl;
    
    std::vector<cv::Mat> retval2;

    for(auto i = start; i < end; i++){
        retval2.push_back(baggify(i->first, i->second, frameVocab));
    }
    return retval2;
}

template<typename IndexIt>
std::vector<cv::Mat> flatScenesBags(IVideo& video, IndexIt start, IndexIt end, cv::Mat frameVocab){
    static_assert(is_pair_iterator_v<IndexIt>, 
        "flatScenesBags requires an iterator to a pair");
    
    auto accessor = [](const Frame& frame) { return frame.descriptors; };
    auto& frames = video.frames();
    std::vector<cv::Mat> matrices;
    auto s = matrices.begin();

    std::transform(frames.begin(), frames.end(), std::back_inserter(matrices), accessor);

    std::vector<std::pair<
        decltype(s), 
        decltype(s)>> tran;

    std::transform(start, end, std::back_inserter(tran),
    [&s](auto i){ return std::make_pair(s + i.first,
        s + i.second); });

    return flatScenesBags(tran.begin(), tran.end(), frameVocab);
}

template<typename Cmp>
std::vector<cv::Mat> flatScenesBags(IVideo &video, Cmp comp, double threshold, cv::Mat frameVocab) {
    auto ss = flatScenes(video, comp, threshold);
    return flatScenesBags(video, ss.begin(), ss.end(), frameVocab);
}

void visualizeSubset(std::string fname, const std::vector<int>& subset = {});

template<typename RangeIt,
        typename = std::enable_if_t<is_pair_iterator_v<RangeIt>>,
        typename = void> 
void visualizeSubset(std::string fname, RangeIt begin, RangeIt end) {
    std::vector<int> subset;
    for(auto i = begin; i < end; i++) 
        for(auto j = begin->first; j < begin->second; j++) 
        subset.push_back(j);

    visualizeSubset(fname, subset);
}

template<typename It,
        typename = std::enable_if_t<!is_pair_iterator_v<It>>>
void visualizeSubset(std::string fname, It begin, It end) {
    auto size = std::distance(begin, end);
    std::cout << "In visualise subset" << std::endl;

    using namespace cv;

    namedWindow("Display window", WINDOW_NORMAL );

    VideoCapture cap(fname, CAP_ANY);

    int count = -1;
    int index = 0;
    Mat image;

    while(cap.read(image)){
        ++count;
        if(size && count != begin[index]){
            continue;
        }

        index++;
        if(index >= size){
            index = 0;
        }
        imshow("Display window", image);
        waitKey(0);
    };
    destroyWindow("Display window");
}


inline void visualizeSubset(std::string fname, const std::vector<int>& subset){
    visualizeSubset(fname, subset.begin(), subset.end());
}

#endif
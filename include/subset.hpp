#ifndef SUBSET_HPP
#define SUBSET_HPP
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

void visualizeSubset(std::string fname, const std::vector<int>& subset = {});

template<typename RangeIt>
std::enable_if_t<is_pair_iterator_v<RangeIt>, void>
visualizeSubset(std::string fname, RangeIt begin, RangeIt end) {
    std::vector<int> subset;
    for(auto i = begin; i < end; i++)
        for(auto j = begin->first; j < begin->second; j++)
        subset.push_back(j);

    visualizeSubset(fname, subset);
}

template<typename It>
std::enable_if_t<!is_pair_iterator_v<It>, void>
visualizeSubset(std::string fname, It begin, It end) {
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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "instrumentation.hpp"
#include <experimental/filesystem>

template <class T, class RankType>
struct sortable{
    RankType rank;
    T data;
    bool operator<(const sortable& a) const {  return rank < a.rank; }; 
};

std::vector<int> flatScenes(IVideo& video, std::function<cv::Mat(Frame)> extractor, double threshold){
    std::cout << "In flatScenes" << std::endl;

    std::vector<int> retval;    

    auto frames = video.frames();
    if(!frames.size()){
        return retval;
    }

    retval.push_back(0);

    for(int i = 1; i < frames.size(); i++) {
        if(double fs = frameSimilarity(frames[i], frames[i - 1], extractor) < threshold){
            auto b = extractor(frames[i]);
            if(b.dot(b) > .000001){ // do not include black frames
                retval.push_back(i);
            }
        }
    }
    
    return retval;
}

void visualizeSubset(std::string fname, std::vector<int> subset = {}){
    std::sort(subset.begin(), subset.end());
    std::cout << "In visualise subset" << std::endl;

    using namespace cv;
    using namespace cv::xfeatures2d;

    namedWindow("Display window", WINDOW_NORMAL );

    VideoCapture cap(fname, CAP_ANY);

    int count = -1;
    int index = 0;
    Mat image;

    while(cap.read(image)){
        ++count;
        if(subset.size() && count != subset[index]){
            continue;
        }

        index++;
        if(index >= subset.size()){
            index = 0;
        }
        imshow("Display window", image);
        waitKey(0);
    };
    destroyWindow("Display window");
}

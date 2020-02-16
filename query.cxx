#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "instrumentation.hpp"
#include <experimental/filesystem>
#include "sw.hpp"

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

bool file_exists(const string& fname){
  return fs::exists(fname);
}

int main(int argc, char** argv )
{
    std::string s1 = "hello there";
    std::string s2 = "I said hello";

    std::vector<char> v1(s1.begin(), s1.end());
    std::vector<char> v2(s2.begin(), s2.end());

    std::function comp = [](char c1, char c2) {
        return (c1 == c2) * 6 - 3;
    };

    int gapScore = 2;
    int threshold = 3;
    
    auto as = calculateAlignment(v1, v2, comp, threshold, gapScore);
    std::cout << as.size() << std::endl << std::endl;
    std::cout << s1 << std::endl << s2 << std::endl;
    for(auto& a : as){
        std::cout << (std::string)a << std::endl;
    }


    if ( argc < 3 )
    {
        printf("usage: ./main <Database_Path> <BOW_matrix_path>\n");
        return -1;
    }

    if ( !file_exists(argv[2]) ){
        namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

        Mat vocab = constructVocabulary(argv[1], 200, 10);

        cv::FileStorage file(argv[2], cv::FileStorage::WRITE);

        file << "Vocabulary" << vocab;
        file.release();
    }

    Mat myvocab;

    cv::FileStorage fs(argv[2], FileStorage::READ);

    fs["Vocabulary"] >> myvocab;
    fs.release();

    // ==============================
    FileDatabase fd(argv[1]);

    auto videopaths = fd.listVideos();
    bool first = 1;
    Frame firstFrame;

    auto extractor = [&myvocab](Frame f) { return baggify(f, myvocab); };
    auto mycomp = [extractor](Frame f1, Frame f2) { return frameSimilarity(f1, f2, extractor); };

    // for each video, compare first frame to rest of frames
    /*for(auto videopath : videopaths){

        auto video = fd.loadVideo(videopath);
        auto frames = video->frames();
        for(auto& frame : frames){
            if(first){
                first = 0;
                firstFrame = frame;
                std::cout << frameSimilarity(firstFrame, firstFrame, myvocab) << std::endl;
                continue;
            }
            else{
                std::cout << frameSimilarity(firstFrame, frame, myvocab) << std::endl;
            }

            //Mat mymat = baggify(frame, myvocab);
        }
    }*/

    // similarity between each two videos
    for(int i = 0; i < videopaths.size() - 1; i++){
        auto s1 = videopaths[i];
        auto v1 = fd.loadVideo(s1);

        VideoMatchingInstrumenter instrumenter(*v1);
        auto reporter = getReporter(instrumenter);
        for(int j = i + 1; j < videopaths.size(); j++){
            auto s2 = videopaths[j];
            std::cout << "Comparing " << s1 << " to " << s2 << std::endl;
            auto v2 = fd.loadVideo(s2);
            std::cout << "Similarity: " << boneheadedSimilarity(*v1, *v2, mycomp, reporter) << std::endl << std::endl;
        }

        EmmaExporter().exportTimeseries(s1, "frame no.", "cosine distance", instrumenter.getTimeSeries());
    }   

    return 0;
}

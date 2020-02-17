#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "instrumentation.hpp"
#include <experimental/filesystem>
#include "sw.hpp"
#include "keyframes.hpp"
#include "kmeans2.hpp"
#include <chrono>


namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace chrono;

bool file_exists(const string& fname){
  return fs::exists(fname);
}

int main(int argc, char** argv )
{
    //srand(time(0));    
    srand(500);

    cv::Mat randm(100, 2, CV_32F);
    for(int i = 0; i < randm.rows; i++){
        for(int j = 0; j < randm.cols; j++){
            randm.at<float>(i, j) = ((float)rand())/RAND_MAX;
        }
    }

    auto start = high_resolution_clock::now(); 
    cv::Mat centers = kmeans2(randm, 3, 1);
    auto stop = high_resolution_clock::now(); 

    auto duration = duration_cast<seconds>(stop - start); 

    std::cout << "Kmeans took " << duration.count() << " seconds" << std::endl;

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


        std::cout << "About to start old kmeans" << std::endl;
        auto start = high_resolution_clock::now();
        Mat vocab = constructVocabulary(argv[1], 200, 10);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);
        std::cout << "Old constructVocabulary took " << duration.count() << " seconds (3 attempts)" << std::endl;

        cv::FileStorage file(argv[2], cv::FileStorage::WRITE);

        file << "Vocabulary" << vocab;
        file.release();

        std::cout << "About to start new kmeans" << std::endl;
        start = high_resolution_clock::now();
        Mat vocab2 = constructMyVocabulary(argv[1], 200, 10);
        stop = high_resolution_clock::now();
        duration = duration_cast<seconds>(stop - start);
        std::cout << "constructMyVocabulary took " << duration.count() << " seconds (1 attempt)" << std::endl;

        cv::FileStorage file2("dump_mykmeans", cv::FileStorage::WRITE);

        file2 << "Vocabulary" << vocab2;
        file2.release();
    }

    Mat myvocab;

    cv::FileStorage fs(argv[2], FileStorage::READ);

    fs["Vocabulary"] >> myvocab;
    fs.release();

    // ==============================

    {
        // key frame stuff
        std::cout << "Key frame stuff" << std::endl;
        auto extractor = [&myvocab](Frame f) { return baggify(f, myvocab); };
        FileDatabase fd(argv[1]);

        auto videopaths = fd.listVideos();
        std::cout << "Got video path list" << std::endl;

        for(auto& vp : videopaths){
            auto vid = fd.loadVideo(vp);
            auto ss = flatScenes(*vid, extractor, .5);
            std::cout << "Video: " << vp << ", scenes: " << ss.size() << std::endl;
            for(auto& a : ss){
                std::cout << a << ", ";
            }
            std::cout << std::endl;

            visualizeSubset(vp, ss);
        }
    }

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

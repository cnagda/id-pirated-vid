#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "instrumentation.hpp"
#include <experimental/filesystem>
#include "sw.hpp"
#include "keyframes.hpp"
#include "kmeans2.hpp"


namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;

bool file_exists(const string& fname){
  return fs::exists(fname);
}




int main(int argc, char** argv )
{
    if ( argc < 4 )
    {
        printf("usage: ./main <Database_Path> <BOW_matrix_path> <Frame_vocab_matrix_path>\n");
        return -1;
    }

    if ( !file_exists(argv[2]) ){
        namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

        FileDatabase db(argv[1]);
        cv::Mat descriptors;
        for(auto &video : db.listVideos())
            for(auto &&frame : db.loadVideo(video)->frames())
                descriptors.push_back(frame.descriptors);

        std::cout << "About to start old kmeans" << std::endl;
        Mat vocab = constructVocabulary(descriptors, 2000);
        std::cout << "Done with kmeans" << std::endl;

        cv::FileStorage file(argv[2], cv::FileStorage::WRITE);

        file << "Vocabulary" << vocab;
        file.release();
    }

    Mat myvocab;

    cv::FileStorage fs(argv[2], FileStorage::READ);

    fs["Vocabulary"] >> myvocab;
    fs.release();

    Mat myframevocab;

    cv::FileStorage fs2(argv[3], FileStorage::READ);

    fs2["Frame_Vocabulary"] >> myframevocab;
    fs2.release();

    FileDatabase fd(argv[1]);

    auto videopaths = fd.listVideos();
    bool first = 1;
    Frame firstFrame;

    auto mycomp = BOWComparator(myvocab);

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

    std::function intcomp = [](cv::Mat b1, cv::Mat b2) { 
        auto a = cosineSimilarity(b1, b2);
        return a > 0.8 ? 3 : -3; 
    };

    // similarity between each two videos
    for(int i = 0; i < videopaths.size() - 1; i++){
        auto s1 = videopaths[i];
        auto v1 = fd.loadVideo(s1);
        
        auto ss = flatScenes(*v1, mycomp, .2);
        auto fb1 = flatScenesBags(ss.begin(), ss.end(), myframevocab);

        VideoMatchingInstrumenter instrumenter(*v1);
        auto reporter = getReporter(instrumenter);
        for(int j = i + 1; j < videopaths.size(); j++){
            auto s2 = videopaths[j];
            std::cout << "Comparing " << s1 << " to " << s2 << std::endl;
            auto v2 = fd.loadVideo(s2);
            std::cout << "Boneheaded Similarity: " << boneheadedSimilarity(*v1, *v2, mycomp, reporter) << std::endl << std::endl;

            auto ss = flatScenes(*v2, mycomp, .2);
            auto fb2 = flatScenesBags(ss.begin(), ss.end(), myframevocab);

            std::cout << "fb1 size: " << fb1.size() << " fb2: " << fb2.size() << std::endl;        
            auto&& alignments = calculateAlignment(fb1, fb2, intcomp, 0, 2);
            
            std::cout << "Scene sw: " << std::endl;
            for(auto& al : alignments){
                std::cout << static_cast<std::string>(al) << std::endl;
            }


        }

        EmmaExporter().exportTimeseries(s1, "frame no.", "cosine distance", instrumenter.getTimeSeries());
    }   

    return 0;
}

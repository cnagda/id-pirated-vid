#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <sys/stat.h>
#include "database.hpp"
#include "bow.hpp"
#include <sys/stat.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

bool file_exists(string fname){
  struct stat buf;   
  return (stat(fname.c_str(), &buf) == 0); 
}

int main(int argc, char** argv )
{
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

    for(auto videopath : videopaths){

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
    }
   
    return 0;
}

#include "keyframes.hpp"
#include "database.hpp"
#include <chrono>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace chrono;

bool file_exists(const string& fname){
  return fs::exists(fname);
}

int main(int argc, char** argv) {
    if ( argc < 3 )
    {
        printf("usage: ./main <Database_Path> <BOW_matrix_path> [Frame_vocabulary_matrix_path]\n");
        return -1;
    }

    if ( !file_exists(argv[2]) ){
        Mat vocab = constructVocabulary(argv[1], 200, 10);

        cv::FileStorage file(argv[2], cv::FileStorage::WRITE);

        file << "Vocabulary" << vocab;
        file.release();
    }

    if ( argc >= 4 ){
        if ( !file_exists(argv[3]) ){
            Mat frameVocab = constructFrameVocabulary();

            cv::FileStorage file(argv[3], cv::FileStorage::WRITE);

            file << "Vocabulary" << vocab;
            file.release();

        }


    }

    namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

    Mat myvocab;

    cv::FileStorage fs(argv[2], FileStorage::READ);

    fs["Vocabulary"] >> myvocab;
    fs.release();

    // key frame stuff
    std::cout << "Key frame stuff" << std::endl;
    auto extractor = [&myvocab](Frame f) { return baggify(f, myvocab); };
    FileDatabase fd(argv[1]);

    auto videopaths = fd.listVideos();
    std::cout << "Got video path list" << std::endl;

    for(auto& vp : videopaths){
        auto vid = fd.loadVideo(vp);
        auto ss = flatScenes(*vid, extractor, .15);
        std::cout << "Video: " << vp << ", scenes: " << ss.size() << std::endl;
        for(auto& a : ss){
            std::cout << a << ", ";
        }
        std::cout << std::endl;

        visualizeSubset(vp, ss);
    }
    return 0;
}

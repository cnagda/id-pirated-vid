#include "keyframes.hpp"
#include "database.hpp"
#include <chrono>

namespace fs = std::experimental::filesystem;
using namespace cv;
using namespace std;
using namespace chrono;

bool file_exists(const string& fname){
  return fs::exists(fname);
}

int main(int argc, char** argv) {
    if ( argc < 3 )
    {
        printf("usage: ./keyframes <Database_Path> <BOW_matrix_path> [Frame_vocabulary_matrix_path]\n");
        return -1;
    }

    if ( !file_exists(argv[2]) ){
        std::cout << "Calculating SIFT vocab" << std::endl;

        FileDatabase db(argv[1]);
        cv::Mat descriptors;
        for(auto &video : db.listVideos())
            for(auto &&frame : db.loadVideo(video)->frames())
                descriptors.push_back(frame.descriptors);

        Mat vocab = constructVocabulary(descriptors, 2000);

        cv::FileStorage file(argv[2], cv::FileStorage::WRITE);

        file << "Vocabulary" << vocab;
        file.release();
    }

    namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.

    Mat myvocab;

    cv::FileStorage fs(argv[2], FileStorage::READ);

    fs["Vocabulary"] >> myvocab;
    fs.release();

    if ( argc >= 4 ){
        if ( !file_exists(argv[3]) ){
            std::cout << "Calculating frame vocab" << std::endl;

            FileDatabase db(argv[1]);
            cv::Mat descriptors;
            for(auto &video : db.listVideos())
                for(auto &&frame : db.loadVideo(video)->frames())
                    descriptors.push_back(baggify(frame.descriptors, myvocab));

            Mat frameVocab = constructVocabulary(descriptors, 200);

            cv::FileStorage file(argv[3], cv::FileStorage::WRITE);

            file << "Frame_Vocabulary" << frameVocab;
            file.release();
        }
    }

    // key frame stuff
    std::cout << "Key frame stuff" << std::endl;
    auto comp = BOWComparator(myvocab);
    FileDatabase fd(argv[1]);

    auto videopaths = fd.listVideos();
    std::cout << "Got video path list" << std::endl;

    for(auto& vp : videopaths){
        auto vid = fd.loadVideo(vp);
        auto ss = flatScenes(*vid, comp, .15);
        std::cout << "Video: " << vp << ", scenes: " << ss.size() << std::endl;
        for(auto& a : ss){
            std::cout << a.first << ", ";
        }
        std::cout << std::endl;

        visualizeSubset(vp, ss.begin(), ss.end());
    }
    return 0;
}

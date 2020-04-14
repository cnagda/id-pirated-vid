#include <iostream>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include <experimental/filesystem>

#define DBPATH      1
#define VIDPATH     2

namespace fs = std::experimental::filesystem;
using namespace std;

bool file_exists(const string& fname){
  return fs::exists(fname);
}




int main(int argc, char** argv )
{

    bool DEBUG = false;
    if ( argc < 3 )
    {
        // TODO: better args parsing
        printf("usage: ./query <Database_Path> <test_video>\n");
        return -1;
    }

    auto& fd = *query_database_factory(argv[DBPATH], -1, -1, -1).release();
    std::string videoname = fs::path(argv[VIDPATH]).filename().string();
    auto video2 = make_query_adapter(fd, getSIFTVideo(argv[VIDPATH]), "totallydifferenttestvid.mp4");
    auto match = findMatch(video2, fd);
    if(match) {
        std::cout << std::endl << "confidence: " << match->matchConfidence << " video: " << match->video << std::endl<<std::endl;
        int count = 0;
        for (auto a : match->alignments) {
            if (a.score > (match->matchConfidence / 2.5)) {
                count++;
                std::cout << "Alignment " << count << ", Score: " << a.score << std::endl;
                std::cout << "Scene range in " << match->video << ": [" << a.startKnown << ", " << a.endKnown << ")" << std::endl;
                std::cout << "Scene range in " << videoname << ": [" << a.startUnknown << ", " << a.endUnknown << ")" << std::endl<<std::endl;
            }
        }
    }

    // namedWindow("Display window", WINDOW_NORMAL );// Create a window for display.
    //
    // auto myvocab = loadVocabulary<Vocab<IScene>>(fd)->descriptors();
    // auto myframevocab = loadVocabulary<Vocab<Frame>>(fd)->descriptors();
    //
    // auto videopaths = fd.loadVideo();
    // bool first = 1;
    // Frame firstFrame;
    //
    // auto mycomp = BOWComparator(myvocab);
    //
    // // for each video, compare first frame to rest of frames
    // /*for(auto videopath : videopaths){
    //
    //     auto video = fd.loadVideo(videopath);
    //     auto frames = video->frames();
    //     for(auto& frame : frames){
    //         if(first){
    //             first = 0;
    //             firstFrame = frame;
    //             std::cout << frameSimilarity(firstFrame, firstFrame, myvocab) << std::endl;
    //             continue;
    //         }
    //         else{
    //             std::cout << frameSimilarity(firstFrame, frame, myvocab) << std::endl;
    //         }
    //
    //         //Mat mymat = baggify(frame, myvocab);
    //     }
    // }*/
    //
    // auto intcomp = [](auto& b1, auto& b2) {
    //     auto a = cosineSimilarity(b1, b2);
    //     return a > 0.8 ? 3 : -3;
    // };
    //
    // // similarity between each two videos
    // for(int i = 0; i < videopaths.size() - 1; i++){
    //     auto& v1 = videopaths[i];
    //
    //     auto fb1 = flatScenesBags(*v1, mycomp, 0.2, myframevocab);
    //
    //     VideoMatchingInstrumenter instrumenter(*v1);
    //     auto reporter = getReporter(instrumenter);
    //     for(int j = i + 1; j < videopaths.size(); j++){
    //         auto& v2 = videopaths[j];
    //         std::cout << "Comparing " << v1->name << " to " << v2->name << std::endl;
    //
    //         std::cout << "Boneheaded Similarity: " << boneheadedSimilarity(*v1, *v2, mycomp, reporter) << std::endl << std::endl;
    //
    //         auto fb2 = flatScenesBags(*v2, mycomp, 0.2, myframevocab);
    //
    //         std::cout << "fb1 size: " << fb1.size() << " fb2: " << fb2.size() << std::endl;
    //         auto&& alignments = calculateAlignment(fb1, fb2, intcomp, 0, 2);
    //
    //         std::cout << "Scene sw: " << std::endl;
    //         for(auto& al : alignments){
    //             std::cout << static_cast<std::string>(al) << std::endl;
    //         }
    //
    //
    //     }
    //
    //     EmmaExporter().exportTimeseries(v1->name, "frame no.", "cosine distance", instrumenter.getTimeSeries());
    // }

    return 0;
}

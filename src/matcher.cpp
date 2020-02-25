#include <opencv2/features2d.hpp>
#include <fstream>
#include <iostream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"
#include "kmeans2.hpp"
#include "sw.hpp"

// bag of frames
cv::Mat constructFrameVocabulary(const std::string& path, cv::Mat vocab, int K, int speedinator, cv::Mat centers, bool online){
    FileDatabase fd(path);

	auto videopaths = fd.listVideos();
	std::vector<cv::Mat> allFeatures;

	// collect all features
	for(auto videopath : videopaths){

        int count = 0;

		auto video = fd.loadVideo(videopath);
		auto frames = video->frames();
		for(auto& frame : frames){
            count++;
            // construct vocab based on only one out of every speedinator frames
            if(count % speedinator) continue;

            // won't be able to vconcat if cols are different sizes
            if (frame.descriptors.cols == 0) continue;

            if(frame.descriptors.rows == 0) continue;

			//allFeatures.push_back(frame.descriptors.clone());
            allFeatures.push_back(baggify(frame.descriptors, vocab));
		}
	}

    std::cout << "Collected: " << allFeatures.size() << std::endl;

    // matrix containing all collected bags of words
	cv::Mat descriptors;
	cv::vconcat(allFeatures, descriptors);

    std::cout << "Concatenated: " << descriptors.rows << std::endl;

	K = (K == -1)? (descriptors.rows / 20) : K;
	//cv::BOWKMeansTrainer trainer(K);
    
    cv::Mat retval;

    kmeans(descriptors, K, centers, cv::TermCriteria(), 1, online? cv::KMEANS_USE_INITIAL_LABELS : cv::KMEANS_PP_CENTERS, retval);

    std::cout << "About to return" << std::endl;

    return retval;
}

cv::Mat constructMyVocabulary(const std::string& path, int K, int speedinator){
	FileDatabase fd(path);

	auto videopaths = fd.listVideos();
	std::vector<cv::Mat> allFeatures;

	// collect all features
	for(auto videopath : videopaths){

        int count = 0;

		auto video = fd.loadVideo(videopath);
		auto frames = video->frames();
		for(auto& frame : frames){
            count++;
            // construct vocab based on only one out of every speedinator frames
            if(count % speedinator) continue;

            // won't be able to vconcat if cols are different sizes
            if (frame.descriptors.cols == 0) continue;

			allFeatures.push_back(frame.descriptors.clone());
		}
	}

    std::cout << "Collected: " << allFeatures.size() << std::endl;

    // for(int i = 0; i < 30; i++){
    //     auto& af = allFeatures;
    //     std::cout << af[i].rows << " X " << af[i].cols << std::endl;
    // }


	cv::Mat descriptors;
	cv::vconcat(allFeatures, descriptors);

    std::cout << "Concatenated" << std::endl;

	K = (K == -1)? (descriptors.rows / 20) : K;
    return kmeans2(descriptors, K, 1);
}

double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter){
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    int len = std::min(frames1.size(), frames2.size());

    for(int i = 0; i < len; i++){
        auto t = comparator(frames1[i], frames2[i]);
        if(reporter) reporter(FrameSimilarityInfo{t, frames1[i], frames2[i], i, i, &v1, &v2});

        total += (t != -1)? t : 0;
    }

    return total/len;
}

std::optional<MatchInfo> findMatch(IVideo& target, IDatabase& db, cv::Mat vocab) {
    auto videopaths = db.listVideos();

    VideoMatchingInstrumenter instrumenter(target);
    auto reporter = getReporter(instrumenter);
    auto extractor = [&vocab](Frame f) { return baggify(f.descriptors, vocab); };
    auto mycomp = [extractor](Frame f1, Frame f2) { return frameSimilarity(f1, f2, extractor); };
    std::function intcomp = [mycomp](Frame f1, Frame f2) { return mycomp(f1, f2) > 0.8 ? 3 : -3; };

    MatchInfo match;

    for(auto s2 : videopaths) {
        auto v2 = db.loadVideo(s2);
        std::cout << "Calculating match for " << v2->name << std::endl;

        double score = boneheadedSimilarity(target, *v2, mycomp, reporter);
        if(score > match.matchConfidence) {
            match = MatchInfo{score, 0, 0, v2->name};
        }
        /* auto&& alignments = calculateAlignment(target.frames(), v2->frames(), intcomp, 20, 2);
        if(alignments.size() > 0) {
            auto& a = alignments[0];
            if(a.score > match.matchConfidence) {
                match = MatchInfo{a.score, a.startKnown, a.endKnown, v2->name};
            }
        } */
        
    }

    EmmaExporter().exportTimeseries(target.name, "frame no.", "cosine distance", instrumenter.getTimeSeries());
    
    if(match.matchConfidence > 0.5) {
        return match;
    }

    return std::nullopt;
}

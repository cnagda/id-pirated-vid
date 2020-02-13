#include <opencv2/features2d.hpp>
#include <fstream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"

cv::Mat constructVocabulary(const std::string& path, int K, int speedinator){
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
	cv::BOWKMeansTrainer trainer(K);

    std::cout << "About to return" << std::endl;

	return trainer.cluster(descriptors);
}

cv::Mat baggify(Frame f, cv::Mat vocab){
    cv::BOWImgDescriptorExtractor extractor(cv::FlannBasedMatcher::create());

    extractor.setVocabulary(vocab);

    cv::Mat output;


    if(f.descriptors.rows){
        extractor.compute(f.descriptors, output);
    }
    else{
        //std::cerr << "In baggify: Frame has no key points" << std::endl;
    }

    return output;
}

double frameSimilarity(Frame f1, Frame f2, std::function<cv::Mat(Frame)> extractor){
    auto b1 = extractor(f1), b2 = extractor(f2);

    if(b1.size() != b2.size()) return -1;

    auto b1n = b1.dot(b1);
    auto b2n = b2.dot(b2);

    return b1.dot(b2)/(sqrt(b1n * b2n) + 1e-10);
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
    auto extractor = [&vocab](Frame f) { return baggify(f, vocab); };
    auto mycomp = [extractor](Frame f1, Frame f2) { return frameSimilarity(f1, f2, extractor); };

    MatchInfo match;

    for(auto s2 : videopaths) {
        auto v2 = db.loadVideo(s2);
        double score = boneheadedSimilarity(target, *v2, mycomp, reporter);
        if(score > match.matchConfidence) {
            match = MatchInfo{score, 0, 0, v2.get()};
        }
    }

    EmmaExporter().exportTimeseries(target.name, "frame no.", "cosine distance", instrumenter.getTimeSeries());
    
    if(match.matchConfidence > 0.5) {
        return match;
    }

    return std::nullopt;
}
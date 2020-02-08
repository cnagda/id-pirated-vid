#include <opencv2/features2d.hpp>
#include "database.hpp"

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
            if(count % speedinator) continue;

			allFeatures.push_back(frame.descriptors.clone());
		}
	}

    std::cout << "Collected: " << allFeatures.size() << std::endl;

    for(int i = 0; i < 10; i++){
        auto& af = allFeatures;
        std::cout << af[i].rows << " X " << af[i].cols << std::endl;
    }


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

    extractor.compute(f.descriptors, output);

    return output;
}

double frameSimilarity(Frame f1, Frame f2, cv::Mat vocab){
    auto b1 = baggify(f1, vocab), b2 = baggify(f2, vocab);
    auto b1n = sqrt(b1.dot(b1));
    auto b2n = sqrt(b2.dot(b2));

    return b1.dot(b2)/(b1n * b2n);
}

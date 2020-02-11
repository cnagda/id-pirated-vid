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
            // construct vocab based on only one out of every speedinator frames
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

double boneheadedSimilarity(IVideo& v1, IVideo& v2, std::function<double(Frame, Frame)> comparator){
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    int len = std::min(frames1.size(), frames2.size());

    for(int i = 0; i < len; i++){
        auto t = comparator(frames1[i], frames2[i]);
        std::cout << "f_sim " << t << std::endl;
        total += (t != -1)? t : 0;
    }
    
    return total/len;
}

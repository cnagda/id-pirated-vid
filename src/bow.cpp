#include <opencv2/features2d.hpp>
#include <fstream>
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

double frameSimilarity(Frame f1, Frame f2, cv::Mat vocab){
    auto b1 = baggify(f1, vocab), b2 = baggify(f2, vocab);
    auto b1n = sqrt(b1.dot(b1));
    auto b2n = sqrt(b2.dot(b2));

    auto denom = b1n * b2n;

    if(!denom) return -1;

    return b1.dot(b2)/denom;
}

double boneheadedSimilarity(IVideo& v1, IVideo& v2, cv::Mat vocab){
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    int len = std::min(frames1.size(), frames2.size());

    std::ofstream ofile;
    ofile.open("temp.txt");

    for(int i = 0; i < len; i++){
        auto t = frameSimilarity(frames1[i], frames2[i], vocab);
        if (t != -1)
            ofile << t << "\n";
        total += (t != -1)? t : 0;
    }

    ofile.close();

    return total/len;
}

cv::Mat constructVocabulary(const std::string& path, int K = -1, int speedinator = 1);
cv::Mat baggify(Frame f, cv::Mat vocab);
double frameSimilarity(Frame f1, Frame f2, cv::Mat vocab);
double boneheadedSimilarity(IVideo& v1, IVideo& v2, cv::Mat vocab);

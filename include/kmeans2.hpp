#include <opencv2/opencv.hpp>

cv::Mat kmeans2(cv::Mat input, int K, int attempts, float halt = .02, int epochs = 1000);
cv::Mat kmeans2(cv::Mat input, int K, int attempts, cv::Mat centers);

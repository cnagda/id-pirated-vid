#include "vocabulary.hpp"
#include <fstream>
#include <random>

int findIndex(const cv::Mat& vocab, cv::Point2f point) {
    int minArg = -1;
    float minDistance = std::numeric_limits<float>::max();
    for(int i = 0; i < vocab.rows; i++) {
        float x = vocab.at<float>(i, 0);
        float y = vocab.at<float>(i, 1);
        float distance = sqrt(pow(x - point.x, 2) + pow(y - point.y, 2));
        if(distance < minDistance) {
            minDistance = distance;
            minArg = i;
        }
    }

    return minArg;
}

int main(int argc, char ** argv) {
    if(argc < 2) {
        std::cout << "Usage: ./visualize <num_points>" << std::endl;
        return -1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution d(50.0, 15.0);

    int numPoints = std::stoi(argv[1]);
    int fourth = numPoints / 4;
    std::vector<cv::Point2f> points(numPoints);
    std::generate_n(points.begin(), fourth, [&](){ return cv::Point2f{d(gen), d(gen)}; });
    std::generate_n(points.begin() + fourth + 1, fourth, [&](){ return cv::Point2f{d(gen) - 50, d(gen)}; });
    std::generate_n(points.begin() + 2 * fourth + 1, fourth, [&](){ return cv::Point2f{d(gen), d(gen) - 50}; });
    std::generate(points.begin() + 3 * fourth + 1, points.end(), [&](){ return cv::Point2f{d(gen) - 50, d(gen) - 50}; });

    auto vocab = constructDemoVocabularyHierarchical(points, 3, 50);

    {
        std::ofstream vocabOut("vocab.mat");
        for(int i = 0; i < vocab.rows; i++) {
            vocabOut << vocab.at<float>(i, 0) << " " << vocab.at<float>(i, 1) << std::endl;
        }
    }

    std::ofstream out("visualize.mat");
    for(auto point : points) {
        out << point.x << " " << point.y << " " << findIndex(vocab, point) << std::endl;
    }

    return 0;
}
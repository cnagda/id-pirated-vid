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

    std::uniform_int_distribution d(-100, 100);

    int numPoints = std::stoi(argv[1]);
    std::vector<cv::Point2f> points(numPoints);
    std::generate(points.begin(), points.end(), [&](){ return cv::Point2f{d(gen), d(gen)}; });

    auto vocab = constructDemoVocabularyHierarchical(points, 3, 50);

    std::vector<std::array<int, 3>> matPoints(points.size());
    std::transform(points.begin(), points.end(), matPoints.begin(), 
        [&](auto i) -> std::array<int, 3> {
            return {i.x, i.y, findIndex(vocab, i)};
        });

    points.clear();

    {
        std::ofstream vocabOut("vocab.mat");
        for(int i = 0; i < vocab.rows; i++) {
            vocabOut << vocab.at<float>(i, 0) << " " << vocab.at<float>(i, 1) << std::endl;
        }
    }

    std::ofstream out("visualize.mat");
    for(auto point : matPoints) {
        out << point[0] << " " << point[1] << " " << point[2] << std::endl;
    }

    return 0;
}
#include <iostream>
#include <string>
#include "database.hpp"
#include "matcher.hpp"
#include "sw.hpp"
#include "vocabulary.hpp"
#include "kmeans2.hpp"
#include <chrono>

using namespace std;
using namespace chrono;
using namespace cv;

int main(int argc, char **argv)
{
    //srand(time(0));
    srand(500);

    cv::Mat randm(100000, 128, CV_32F);
    for (int i = 0; i < randm.rows; i++)
    {
        for (int j = 0; j < randm.cols; j++)
        {
            randm.at<float>(i, j) = ((float)rand()) / RAND_MAX;
        }
    }

    int ncenters = 4;

    auto start = high_resolution_clock::now();
    cv::Mat centers = kmeans2(randm, ncenters, 1);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(stop - start);

    std::cout << "normal Kmeans took " << duration.count() << " seconds" << std::endl;

    start = high_resolution_clock::now();
    centers = fastkmeans2(randm, ncenters);
    stop = high_resolution_clock::now();

    duration = duration_cast<seconds>(stop - start);

    std::cout << "fast Kmeans took " << duration.count() << " seconds" << std::endl;

    std::string s1 = "hello there";
    std::string s2 = "I said hello";

    std::vector<char> v1(s1.begin(), s1.end());
    std::vector<char> v2(s2.begin(), s2.end());

    std::function comp = [](char c1, char c2) {
        return (c1 == c2) * 6 - 3;
    };

    int gapScore = 2;
    int threshold = 3;

    auto as = calculateAlignment(v1, v2, comp, threshold, gapScore);
    std::cout << as.size() << std::endl
              << std::endl;
    std::cout << s1 << std::endl
              << s2 << std::endl;
    for (auto &a : as)
    {
        std::cout << (std::string)a << std::endl;
    }

    return 0;
}
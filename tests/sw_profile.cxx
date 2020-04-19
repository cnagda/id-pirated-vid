#include "sw.hpp"
#include <string>
#include <algorithm>
#include <iostream>

int main()
{
    auto randchar = []() -> char {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };
    std::string s1(8192, 0);
    std::string s2(8192, 0);
    std::generate_n(s1.begin(), 8192, randchar);
    std::generate_n(s2.begin(), 8192, randchar);

    auto alignments = calculateAlignment(
        s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b) { return a == b ? 3 : -3; }, 10, 2);

    if (alignments.size() > 0)
    {
        std::cout << alignments[0].score << std::endl;
    }
    return 0;
}
#include <benchmark/benchmark.h>
#include "sw.hpp"
#include <algorithm>

static void BM_SmithWatterman(benchmark::State &state)
{
    auto randchar = []() -> char {
        const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[rand() % max_index];
    };
    for (auto _ : state)
    {
        state.PauseTiming();

        std::string s1(state.range(0), 0);
        std::string s2(state.range(1), 0);
        std::generate_n(s1.begin(), state.range(0), randchar);
        std::generate_n(s2.begin(), state.range(1), randchar);

        state.ResumeTiming();

        auto alignments = calculateAlignment(
            s1.begin(), s1.end(), s2.begin(), s2.end(), [](auto a, auto b) { return a == b ? 3 : -3; }, 10, 2);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_SmithWatterman)->Ranges({{8, 8 << 10}, {8, 8 << 10}});

#ifndef SW_HPP
#define SW_HPP

#include <vector>
#include <utility>
#include <functional>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include "matrix.hpp"
#include "matcher.hpp"
#include <iostream>
#include <atomic>
#include <cassert>
#include <tbb/parallel_do.h>

#define BLOCK_SIZE 128
#define BLOCK_MAX_DIM(max) (((max) + BLOCK_SIZE - 1) / BLOCK_SIZE)

typedef VectorMatrix<uint8_t> SourceMatrix;
typedef Eigen::MatrixXi ScoreMatrix;

template<typename It>
std::vector<ItAlignment<It>> findAlignments(It known, It unknown, ScoreMatrix& matrix, const SourceMatrix& sources, unsigned int maxAlignments) {
    std::vector<ItAlignment<It>> alignments;

    for(unsigned int u; u < maxAlignments; u++){
        auto [i, j] = slowMatrixMax(matrix);

        std::cout << "i: " << i << " j: " << j << std::endl;
        if(i <= 0 || j <= 0) {
            return alignments;
        }
        if(matrix(i, j) <= 0){
            return alignments;
        }

        ItAlignment<It> a;

        a.endUnknown = unknown + i;
        a.endKnown = known + j;
        a.score = matrix(i, j);

        do{
            matrix(i, j) = -1;
            int tempi = i - (bool)sources(i, j);
            j -= sources(i, j) < 2;
            i = tempi;

        } while(matrix(i, j) > 0);
        if(matrix(i, j) == -1) {
            continue;
        }

        a.startUnknown = unknown + i;
        a.startKnown = known + j;
        alignments.push_back(a);
    }

    return alignments;
}


template<typename It, typename Cmp>
void populateSearchSpaceWavefront(It known, It unknown, int m, int n, unsigned int gapScore, Cmp comp, ScoreMatrix& matrix, SourceMatrix& sources) {
    // populate matrix
    typedef std::pair<int, int> block;
    auto block_max_cols = BLOCK_MAX_DIM(n);
    auto block_max_rows = BLOCK_MAX_DIM(m);

    // intel atomics are useless and slow
    VectorMatrix<std::atomic<uint8_t>> counter(block_max_rows, block_max_cols);
    for(int i = 1; i < block_max_rows; i++) {
        for(int j = 1; j < block_max_cols; j++) {
            counter(i, j).store(2, std::memory_order_relaxed);
        }
    }
    for(int i = 1; i < block_max_cols; i++) counter(0, i).store(1, std::memory_order_relaxed);
    for(int i = 1; i < block_max_rows; i++) counter(i, 0).store(1, std::memory_order_relaxed);
    counter(0, 0).store(0, std::memory_order_relaxed);

    auto block_process = [&](const block& origin, tbb::parallel_do_feeder<block>& feeder) {
        auto [bi, bj] = origin;
        auto si = bi * BLOCK_SIZE;
        auto sj = bj * BLOCK_SIZE;
        for(unsigned int i = si + 1; i < std::min(si + BLOCK_SIZE + 1, m); i++){
            for(unsigned int j = sj + 1; j < std::min(sj + BLOCK_SIZE + 1, n); j++){
                // if((i * n + j) % 20 == 0) std::cout << "Inner loop: " << i * n + j << " out of " << m * n << std::endl;
                int max = 0;
                int score = 0;

                // first comparison: west cell (deletion)
                score = matrix(i, j-1) - gapScore;
                if(score > max){
                    max = score;
                    sources(i, j) = 0;
                }

                // second comparison: north cell (insertion)
                score = matrix(i-1, j) - gapScore;
                if(score > max){
                    max = score;
                    sources(i, j) = 2;
                }

                // last comparison: north-west cell (alignment)
                if(max < (score = comp(known[j - 1], unknown[i - 1]) + matrix(i - 1, j - 1))){
                    max = score;
                    sources(i, j) = 1;
                }

                matrix(i, j) = max;
            }
        }
        if( bj+1<block_max_cols && counter(bi, bj+1).fetch_sub(1, std::memory_order_relaxed)==1 ) {
            feeder.add( block(bi,bj+1) );
        }
        if( bi+1<block_max_rows && counter(bi+1, bj).fetch_sub(1, std::memory_order_relaxed)==1 ) {
            feeder.add( block(bi+1,bj) );
        }
    };

    std::array<block, 1> origin{block(0, 0)};
    tbb::parallel_do(origin, block_process);
}

template<typename It, typename Cmp>
void populateSearchSpace(It known, It unknown, int m, int n, unsigned int gapScore, Cmp comp, ScoreMatrix& matrix, SourceMatrix& sources) {
    // populate matrix
    for(int i = 1; i < m; i++){
        for(int j = 1; j < n; j++){
            // if((i * n + j) % 20 == 0) std::cout << "Inner loop: " << i * n + j << " out of " << m * n << std::endl;
            int max = 0;
            int score = 0;

            // first comparison: west cell (deletion)
            score = matrix(i, j-1) - gapScore;
            if(score > max){
                max = score;
                sources(i, j) = 0;
            }

            // second comparison: north cell (insertion)
            score = matrix(i-1, j) - gapScore;
            if(score > max){
                max = score;
                sources(i, j) = 2;
            }

            // last comparison: north-west cell (alignment)
            if(max < (score = comp(known[j - 1], unknown[i - 1]) + matrix(i - 1, j - 1))){
                max = score;
                sources(i, j) = 1;
            }

            matrix(i, j) = max;
        }
    }
}

template <typename It, typename Cmp>
std::vector<ItAlignment<It>> calculateAlignment(It known, It knownEnd, It unknown, It unknownEnd, Cmp comp, unsigned int maxAlignments, unsigned int gapScore){
    int m = std::distance(unknown, unknownEnd) + 1;
    int n = std::distance(known, knownEnd) + 1;

    ScoreMatrix matrix(m, n);
    matrix.col(0).fill(0);
    matrix.row(0).fill(0);
    // 0 for left, 1 for diagonal, 2 for up
    SourceMatrix sources(m, n);

    populateSearchSpaceWavefront(known, unknown, m, n, gapScore, comp, matrix, sources);
    //FIXME
#ifdef SW_COUT
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++)
            std::cout << matrix(i, j) << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif
    
    return findAlignments(known, unknown, matrix, sources, maxAlignments);
}

template <typename Range, typename Cmp>
std::vector<Alignment> calculateAlignment(Range&& known, Range&& unknown, Cmp&& comp, unsigned int maxAlignments, unsigned int gapScore){
    auto alignments = calculateAlignment(known.begin(), known.end(), unknown.begin(), unknown.end(), comp, maxAlignments, gapScore);
    std::vector<Alignment> ret;
    ret.reserve(alignments.size());

    std::transform(alignments.begin(), alignments.end(), std::back_inserter(ret), [&known, &unknown](auto val) {
        return Alignment{
            static_cast<unsigned int>(std::distance(known.begin(), val.startKnown)),
            static_cast<unsigned int>(std::distance(unknown.begin(), val.startUnknown)),
            static_cast<unsigned int>(std::distance(known.begin(), val.endKnown)),
            static_cast<unsigned int>(std::distance(unknown.begin(), val.endUnknown)),
            val.score
        };
    });

    return ret;
}

#endif

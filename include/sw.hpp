#include <vector>
#include <utility>
#include <functional>
#include <iostream>


template<typename It>
struct ItAlignment {
    It startKnown, startUnknown, endKnown, endUnknown;
    int score;
};

struct Alignment : ItAlignment<int> {
    operator std::string(){
        return std::to_string(startKnown) + " " + std::to_string(endKnown) + 
               " " + std::to_string(startUnknown) + " " + std::to_string(endUnknown) + " " +
               std::to_string(score);
    }
};

std::pair<int, int> slowMatrixMax(std::vector<std::vector<int>> & matrix){
    int max = -1, mi = -1, mj = -1;

    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[0].size(); j++){
            if(max < matrix[i][j]){
                max = matrix[i][j];
                mi = i;
                mj = j;
            }
        }
    }

    return std::make_pair(mi, mj);
}

template <typename It, typename Cmp> 
std::vector<ItAlignment<It>> calculateAlignment(It known, It knownEnd, It unknown, It unknownEnd, Cmp comp, int threshold, unsigned int gapScore){
    using Alignment = ItAlignment<It>;

    int m = std::distance(unknown, unknownEnd);
    int n = std::distance(known, knownEnd);

    std::vector<std::vector<int>> matrix(m + 1, std::vector<int>(n + 1, 0));
    // 0 for left, 1 for diagonal, 2 for up
    std::vector<std::vector<int>> sources(m + 1, std::vector<int>(n + 1, -1));

    // populate matrix
    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
            int max = 0;
            int score = 0;

            // first comparison: west cell (deletion)
            score = matrix[i][j-1] - gapScore;
            if(score > max){
                max = score;
                sources[i][j] = 0;
            }
            
            // second comparison: north cell (insertion)
            score = matrix[i-1][j] - gapScore;
            if(score > max){
                max = score;
                sources[i][j] = 2;
            }
            
            // last comparison: north-west cell (alignment)
            if(max < (score = comp(known[j], unknown[i]) + matrix[i - 1][j - 1])){
                max = score;
                sources[i][j] = 1;
            }

            // finished all comparisons
            matrix[i][j] = std::max(0, max);
        }
    }

    //FIXME
    for(auto& a : matrix){
        for(auto& b : a){
            std::cout << std::setw(2) << b << " ";
        }
        std::cout << std::endl;
    }

    std::vector<Alignment> alignments;

    while(1){
        auto [i, j] = slowMatrixMax(matrix);
        if(i <= 0 || j <= 0){
            return alignments;
        }
        if(matrix[i][j] < threshold){
            return alignments;
        }

        Alignment a;

        a.endUnknown = unknown + i;
        a.endKnown = known + j;
        a.score = matrix[i][j];
        
        do{
            if(matrix[i][j] == 0){
                a.startUnknown = unknown + i;
                a.startKnown = known + j;
                alignments.push_back(a);
                break;
            }

            matrix[i][j] = -1;
            int tempi = i - (bool)sources[i][j];
            j -= sources[i][j] < 2;
            i = tempi;

        } while(matrix[i][j] >= 0);
    }
}

template <typename T>
std::vector<Alignment> calculateAlignment(std::vector<T> & known, std::vector<T> & unknown, std::function<int(T, T)> comp, int threshold, unsigned int gapScore){
    auto alignments = calculateAlignment(known.begin(), known.end(), unknown.begin(), unknown.end(), comp, threshold, gapScore);
    std::vector<Alignment> ret;
    ret.reserve(alignments.size());

    std::transform(alignments.begin(), alignments.end(), std::back_inserter(ret), [&known, &unknown](auto val) -> Alignment {
        return Alignment{
            std::distance(known.begin(), val.startKnown),
            std::distance(known.begin(), val.endKnown),
            std::distance(unknown.begin(), val.startUnknown),
            std::distance(unknown.begin(), val.endUnknown),
            val.score
        };
    });

    return ret;    
}

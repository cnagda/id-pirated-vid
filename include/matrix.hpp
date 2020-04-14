#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <utility>
#include <vector>
#include <Eigen/Core>

std::pair<int, int> slowMatrixMax(std::vector<std::vector<int>> & matrix);

template<typename Matrix>
inline std::pair<int, int> slowMatrixMax(Matrix&& matrix) {
    int i, j;
    matrix.maxCoeff(&i, &j);
    return {i, j};
}

#endif
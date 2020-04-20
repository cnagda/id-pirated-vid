#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <utility>
#include <vector>
#include <Eigen/Core>

template <typename T>
class VectorMatrix
{
private:
    std::vector<T> matrix;

public:
    typedef typename std::vector<T>::size_type size_type;
    typedef size_type index_type;

    size_type rows, cols;

    VectorMatrix(size_type rows, size_type cols, const T &initial_value) : rows(rows), cols(cols), matrix(rows * cols, initial_value){};
    VectorMatrix(size_type rows, size_type cols) : rows(rows), cols(cols), matrix(rows * cols){};

    constexpr T &operator()(size_type i, size_type j) &
    {
        return matrix[i * cols + j];
    }
    constexpr const T &operator()(size_type i, size_type j) const &
    {
        return matrix[i * cols + j];
    }

    constexpr auto begin() { return matrix.begin(); }
    constexpr auto end() { return matrix.end(); }
};

std::pair<int, int> slowMatrixMax(std::vector<std::vector<int>> &matrix);

template <typename Matrix>
inline std::pair<int, int> slowMatrixMax(Matrix &&matrix)
{
    int i, j;
    matrix.maxCoeff(&i, &j);
    return {i, j};
}

#endif
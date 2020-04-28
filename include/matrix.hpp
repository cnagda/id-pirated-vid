#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <utility>
#include <vector>

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

template<typename T>
std::pair<int, int> slowMatrixMax(const VectorMatrix<T> &matrix)
{
    int max = -1, mi = -1, mj = -1;

    for (int i = 0; i < matrix.rows; i++)
    {
        for (int j = 0; j < matrix.cols; j++)
        {
            if (max < matrix(i, j))
            {
                max = matrix(i, j);
                mi = i;
                mj = j;
            }
        }
    }

    return {mi, mj};
}

#endif
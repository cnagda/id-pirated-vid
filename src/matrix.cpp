#include "matrix.hpp"

std::pair<int, int> slowMatrixMax(std::vector<std::vector<int>> &matrix)
{
    int max = -1, mi = -1, mj = -1;

    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            if (max < matrix[i][j])
            {
                max = matrix[i][j];
                mi = i;
                mj = j;
            }
        }
    }

    return std::make_pair(mi, mj);
}
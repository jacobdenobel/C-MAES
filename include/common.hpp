#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <algorithm>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Array = Eigen::ArrayXd;

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &x)
{
    for (auto &xi : x)
        os << xi << ' ';
    return os;
}

namespace utils
{
    /**
     * @brief Return an array of indexes of the sorted array
     * sort indexes based on comparing values in v using std::stable_sort instead
     * of std::sort  to avoid unnecessary index re-orderings
     * when v contains elements of equal values.
     *
     * @param v
     * @return std::vector<size_t>
     */
    std::vector<size_t> sort_indexes(const Vector &v)
    {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        std::stable_sort(idx.begin(), idx.end(),
                         [&v](size_t i1, size_t i2)
                         { return v[i1] < v[i2]; });

        return idx;
    }

    /**
     * @brief Concat two matrices inplace, i.e. put Y in X (colwise)
     * 
     * @param X target matrix
     * @param Y source matrix
     */
    inline void hstack(Matrix& X, const Matrix& Y) {
        X.conservativeResize(Eigen::NoChange, X.cols() + Y.cols());
        X.rightCols(Y.cols()) = Y;
    }

    /**
     * @brief Concat two matrices inplace, i.e. put Y in X (rowwise)
     * 
     * @param X target matrix
     * @param Y source matrix
     */
    inline void vstack(Matrix& X, const Matrix& Y) {
        X.conservativeResize(X.rows() + Y.rows(), Eigen::NoChange);
        X.bottomRows(Y.rows()) = Y;
    }

    /**
     * @brief Concat two vectors
     * 
     * @param x target vector
     * @param y source vector
     */
    inline void concat(Vector& x, const Vector& y){
        x.conservativeResize(x.rows() + y.rows(), Eigen::NoChange);
        x.bottomRows(y.rows()) = y;
    }
}

namespace rng
{
    static inline int SEED = 0;
    static inline std::mt19937 GENERATOR(SEED);

    /**
     * @brief Set the global seed and reseed the random generator (mt19937)
     *
     * @param seed
     */
    void set_seed(const int seed)
    {
        SEED = seed;
        GENERATOR.seed(seed);
    }
    /**
     * @brief random integer generator using global GENERATOR
     *
     * @param l lower bound
     * @param h upper bound
     * @return int a random integer
     */
    int random_integer(int l, int h)
    {
        std::uniform_int_distribution<> distrib(l, h);
        return distrib(GENERATOR);
    }
}
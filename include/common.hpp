#pragma once

#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <algorithm>
#include <numeric>
#include <ciso646>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Array = Eigen::ArrayXd;

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &x);

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
    std::vector<size_t> sort_indexes(const Vector &v);

    /**
     * @brief Concat two matrices inplace, i.e. put Y in X (colwise)
     *
     * @param X target matrix
     * @param Y source matrix
     */
    void hstack(Matrix &X, const Matrix &Y);

    /**
     * @brief Concat two matrices inplace, i.e. put Y in X (rowwise)
     *
     * @param X target matrix
     * @param Y source matrix
     */
    void vstack(Matrix &X, const Matrix &Y);

    /**
     * @brief Concat two vectors
     *
     * @param x target vector
     * @param y source vector
     */
    void concat(Vector &x, const Vector &y);
}

namespace rng
{
    //! The global seed value
    extern int SEED;

    //! The random generator
    extern std::mt19937 GENERATOR;

    /**
     * @brief Set the global seed and reseed the random generator (mt19937)
     *
     * @param seed
     */
    void set_seed(const int seed);
    /**
     * @brief random integer generator using global GENERATOR
     *
     * @param l lower bound
     * @param h upper bound
     * @return int a random integer
     */
    int random_integer(int l, int h);
}
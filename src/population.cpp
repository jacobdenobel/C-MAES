#include "population.hpp"

void Population::sort()
{
    const auto idx = utils::sort_indexes(f);
    X = X(Eigen::all, idx).eval();
    Y = Y(Eigen::all, idx).eval();
    f = f(idx).eval();
    s = s(idx).eval();
}

void Population::operator+=(const Population other)
{
    utils::hstack(X, other.X);
    utils::hstack(Y, other.Y);
    utils::hstack(Z, other.Z);
    utils::concat(f, other.f);
    utils::concat(s, other.s);
    n += other.n;
}

void Population::resize_cols(const size_t size)
{
    n = std::min(size, static_cast<size_t>(X.cols()));
    X.conservativeResize(d, n);
    Y.conservativeResize(d, n);
    Z.conservativeResize(d, n);
    f.conservativeResize(n);
    s.conservativeResize(n);
}

std::ostream &operator<<(std::ostream &os, const Population &p)
{
    os
        << "Population"
        << "\nx=\n"
        << p.X
        << "\ny=\n"
        << p.Y
        << "\nf=\n"
        << p.f.transpose();
    // << "\ns=\n" << p.s;
    return os;
}

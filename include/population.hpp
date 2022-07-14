#pragma once

#include "common.hpp"

struct Population
{
    Matrix X;
    Matrix Z;
    Matrix Y;
    Vector f;
    Vector s;

    size_t d;
    size_t n;

    Population(const size_t d, const size_t n)
        : X(d, n), Z(d, n),  Y(d, n), f(n), s(n), d(d), n(n) {}
    
    Population(const Matrix &X, const Matrix &Z, const Matrix &Y, const Vector &f, const Vector &s)
        : X(X), Z(Z), Y(Y), f(f), s(s), d(X.rows()), n(X.cols()) {}

    Population(): Population(0, 0) {}

    void sort()
    {
        const auto idx = utils::sort_indexes(f);
        X = X(Eigen::all, idx).eval();
        Y = Y(Eigen::all, idx).eval();
        f = f(idx).eval();
        s = s(idx).eval();
    }
    
    void operator+=(const Population other){
        utils::hstack(X, other.X);
        utils::hstack(Y, other.Y);
        utils::hstack(Z, other.Z);
        utils::concat(f, other.f);
        utils::concat(s, other.s);
        n += other.n;
    }

    void resize_cols(const size_t size) {
        n = std::min(size, static_cast<size_t>(X.cols()));
        X.conservativeResize(d, n);
        Y.conservativeResize(d, n);
        Z.conservativeResize(d, n);
        f.conservativeResize(n);
        s.conservativeResize(n);
    }

    friend std::ostream& operator<<(std::ostream& os, const Population& dt);
};


std::ostream &operator<<(std::ostream &os, const Population &p)
{
    os  
        << "Population"
        << "\nx=\n" << p.X
        << "\ny=\n" << p.Y
        << "\nf=\n" << p.f.transpose();
        // << "\ns=\n" << p.s;
    return os;
}



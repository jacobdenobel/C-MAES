#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    parameters::Parameters p;
    bool verbose = true;

    ModularCMAES(const parameters::Parameters &p) : p(p) {}

    void mutate(std::function<double(Vector)> objective);

    void select();

    void recombine();

    bool step(std::function<double(Vector)> objective);

    void operator()(std::function<double(Vector)> objective);

    bool sequential_break_conditions(const size_t i, const double f) const;

    bool break_conditions() const;
};

void scale_with_threshold(Matrix& z, const double t);
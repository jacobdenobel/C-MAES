#pragma once

#include "parameters.hpp"

struct ModularCMAES
{
    parameters::Parameters p;
    bool verbose = true;

    ModularCMAES(const parameters::Parameters &p) : p(p) {}

    void recombine();

    bool step(std::function<double(Vector)> objective);

    void operator()(std::function<double(Vector)> objective);

    bool break_conditions() const;
};

void scale_with_threshold(Matrix& z, const double t);
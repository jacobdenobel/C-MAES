#pragma once

#include "common.hpp"
#include "sampling.hpp"
#include "population.hpp"
#include "parameters.hpp"


struct ModularCMAES {
    parameters::Parameters p;
    sampling::Gaussian sampler;


    ModularCMAES(const parameters::Parameters& p): p(p), sampler(p.dim) {

    }

    void mutate(std::function<double(Vector)> objective) {
        for (size_t i = 0; i < p.pop.Z.cols(); ++i)
            p.pop.Z.col(i) = sampler();

        p.pop.Y = p.dyn.B * (p.dyn.d.asDiagonal() * p.pop.Z);
        p.pop.X = (p.dyn.sigma * p.pop.Y).colwise() + p.dyn.m;

        for (size_t i = 0; i < p.pop.X.cols(); ++i){
            p.pop.f(i) = objective(p.pop.X.col(i));
            p.stats.used_budget++;
        }
    } 

    void select() {
        if (p.mod.elitist and p.stats.t != 0)
            p.pop += p.old_pop;

        p.pop.sort();
        p.pop.resize_cols(p.strat.lambda);

        if (p.pop.f(0) < p.stats.fopt){
            p.stats.fopt = p.pop.f(0);
            p.stats.xopt = p.pop.X(Eigen::all, 0);
        }
    }

    void recombine() {
        p.dyn.m_old = p.dyn.m;
        p.dyn.m = p.dyn.m_old + (
            (p.pop.X.leftCols(p.strat.mu).colwise() - p.dyn.m_old) * p.weights.p 
        );
    }

    bool step(std::function<double(Vector)> objective) {
        mutate(objective);
        select();
        recombine();
        p.adapt();
        return !break_conditions();
    }  

    void operator()(std::function<double(Vector)> objective) {
        while(step(objective)) {
            if (p.stats.t % 10 == 0)
                std::cout << p.stats << std::endl;
        }
        std::cout << p.stats << std::endl;
    }

    bool break_conditions() {
        const bool target_reached = p.stats.target >= p.stats.fopt;
        const bool budget_used_up = p.stats.used_budget >= p.stats.budget;
        const bool exceed_gens = p.stats.t >= p.stats.max_generations;

        return exceed_gens or target_reached or budget_used_up;
    }
};




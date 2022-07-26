#include "parameters.hpp"

namespace parameters
{
    Strategy::Strategy(const size_t dim, const Modules& mod, const size_to l, const size_to m)
        : lambda(l.value_or(4 + std::floor(3 * std::log(dim)))),
        mu(m.value_or(lambda / 2)), bounds(bounds::get(dim, mod.bound_correction))
    {
        if (mod.mirrored == sampling::Mirror::PAIRWISE and lambda % 2 != 0)
            lambda++;

        if (mu > lambda)
            mu = lambda / 2;
    }


    Parameters::Parameters(const size_t dim, const Modules& m) : 
        dim(dim), mod(m), dyn(dim), strat(dim, mod),
        weights(dim, strat.mu, strat.lambda, mod),
        pop(dim, strat.lambda),
        old_pop(dim, strat.lambda),        
        sampler(sampling::get(dim, mod, strat)),
        mutation_strategy(mutation::get(mod,           
            strat.mu, weights.mueff,
            static_cast<double>(dim),
            .5 // sigma
        )),
        selection_strategy(std::make_shared<selection::Strategy>(mod)),
        restart_strategy(restart::get(mod.restart_strategy, 
            static_cast<double>(dim), 
            static_cast<double>(strat.lambda), 
            static_cast<double>(strat.mu), 
            stats.budget)
        )
    {
        // Ensure proper initialization of pop.s
        mutation_strategy->sample_sigma(pop);

    }

    Parameters::Parameters(const size_t dim) : Parameters(dim, {}) {}
        
    void Parameters::restart(const std::optional<double>& sigma) {
        weights = Weights(dim, strat.mu, strat.lambda, mod);
        sampler = sampling::get(dim, mod, strat);

        pop = Population(dim, strat.lambda); 
        old_pop = Population(dim, strat.lambda);

        mutation_strategy = mutation::get(mod, strat.mu, weights.mueff,
            static_cast<double>(dim), 
            sigma.value_or(mutation_strategy->sigma0)
        );
        
        mutation_strategy->sample_sigma(pop);
        
        dyn.B = Matrix::Identity(dim, dim);
        dyn.C = Matrix::Identity(dim, dim);
        dyn.inv_root_C = Matrix::Identity(dim, dim);
        dyn.d.setOnes();
        dyn.m = Vector::Random(dim) * 5;
        dyn.m_old.setZero();
        dyn.dm.setZero();
        dyn.pc.setZero();
        dyn.ps.setZero();

    }

    void Parameters::adapt()
    {

        dyn.adapt_evolution_paths(weights, mutation_strategy, stats, strat);
        mutation_strategy->adapt(weights, dyn, pop, old_pop, stats, strat);
        dyn.adapt_covariance_matrix(weights, mod, pop, strat);
        
        if (!dyn.perform_eigendecomposition(stats))
            restart();
        
        old_pop = pop;
        restart_strategy->evaluate(*this);
        
        stats.t++;
    }   
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
    return os
           << "Stats"
           << " t=" << s.t
           << " evals=" << s.evaluations
           << " xopt=("
           << s.xopt.transpose()
           << ") fopt="
           << s.fopt;
}

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


    Parameters::Parameters(const size_t dim) : dim(dim), dyn(dim), strat(dim, mod),
        weights(dim, strat.mu, strat.lambda, mod),
        sampler(sampling::get(dim, mod, strat)),
        mutation_strategy(mutation::get(mod.ssa,
            mod.threshold_convergence, mod.sequential_selection,
            mod.sample_sigma, mod.mirrored, strat.mu, weights.mueff,
            static_cast<double>(dim))
        ),
        selection_strategy(std::make_shared<selection::Strategy>(mod)),
        restart_strategy(restart::get(mod.local_restart, 
            static_cast<double>(dim), 
            static_cast<double>(strat.lambda), 
            static_cast<double>(strat.mu), 
            stats.budget)
        ),
        pop(dim, strat.lambda), old_pop(dim, strat.lambda)
    {
        // Ensure proper initialization of pop.s
        mutation_strategy->ss->sample(dyn.sigma, pop);
    }
        
    void Parameters::restart() {
        weights = Weights(dim, strat.mu, strat.lambda, mod);
        sampler = sampling::get(dim, mod, strat);
        mutation_strategy = mutation::get(mod.ssa,
            mod.threshold_convergence, mod.sequential_selection,
            mod.sample_sigma, mod.mirrored, strat.mu, weights.mueff,
            static_cast<double>(dim)
        );
        pop = Population(dim, strat.lambda);
        old_pop = Population(dim, strat.lambda);

        dyn.restart();
        mutation_strategy->ss->sample(dyn.sigma, pop);
    }

    void Parameters::adapt()
    {

        dyn.adapt_evolution_paths(weights, mutation_strategy, stats, strat);
        mutation_strategy->adapt(weights, dyn, pop, old_pop, stats, strat);
        dyn.adapt_covariance_matrix(weights, mod, pop, strat);
        
        if (!dyn.perform_eigendecomposition(stats))
            restart();

        (*restart_strategy)(*this);
        old_pop = pop;
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

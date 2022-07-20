#include "parameters.hpp"

namespace parameters
{
    Strategy::Strategy(const size_t dim, const Modules &mod, const size_to l, const size_to m)
        : lambda(l.value_or(4 + std::floor(3 * std::log(dim)))),
          mu(m.value_or(lambda / 2)),
          seq_cutoff_factor(std::max(2., seq_cutoff_factor) ? mod.mirrored == Mirrored::PAIRWISE : seq_cutoff_factor),
          seq_cutoff(mu * seq_cutoff_factor), lb(Vector::Ones(dim) * -5.), ub(Vector::Ones(dim) * 5), 
          diameter((ub - lb).norm()),
		  beta(std::log(2.0) / std::max((std::sqrt(dim) * std::log(dim)), 1.0))
    {
        if (mod.mirrored == Mirrored::PAIRWISE and lambda % 2 != 0)
            lambda++;

        if (mu > lambda)
            mu = lambda / 2;
    }

    [[nodiscard]] double Strategy::threshold(const Stats &s) const
    {
        return init_threshold * diameter * pow(static_cast<double>(s.budget - s.evaluations) / static_cast<double>(s.budget), decay_factor);
    }

    Parameters::Parameters(const size_t dim) : dim(dim), dyn(dim), strat(dim, mod), 
                                               weights(dim, strat.mu, strat.lambda, mod),
                                               sampler(get_sampler(dim, mod, strat)),
                                               pop(dim, strat.lambda), old_pop(dim, strat.lambda)
    {
    }

    void Parameters::adapt()
    {
        dyn.adapt(weights, stats, strat, mod, pop);
        old_pop = pop;
        stats.t++;
    }

    std::shared_ptr<sampling::Sampler> Parameters::get_sampler(const size_t dim, const Modules& mod, const Strategy& strat)
    {
        std::shared_ptr<sampling::Sampler> sampler;
    	switch (mod.sampler)
        {
        case BaseSampler::GAUSSIAN:
            sampler = std::make_shared<sampling::Gaussian>(dim);
            break;
        case BaseSampler::SOBOL:
            sampler = std::make_shared<sampling::Sobol>(dim);
            break;
        case BaseSampler::HALTON:
            sampler = std::make_shared<sampling::Halton>(dim);
            break;
        case BaseSampler::TESTER:
            sampler = std::make_shared<sampling::Tester>(dim);
            break;
        };

        auto not_mirrored = mod.mirrored == Mirrored::NONE;
        if (mod.orthogonal)
        {
            auto has_tpa = mod.ssa == StepSizeAdaptation::TPA;
            auto n_samples = std::max(1, (static_cast<int>(strat.lambda) / (2 - not_mirrored)) - (2 * has_tpa));
            sampler = std::make_shared<sampling::Orthogonal>(sampler, n_samples);
        }
        if (not not_mirrored)
            sampler = std::make_shared<sampling::Mirrored>(sampler);
        return sampler;
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

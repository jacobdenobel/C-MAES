#pragma once

#include "population.hpp"
#include "sampling.hpp"
#include "bounds.hpp"
#include "mutation.hpp"
#include "selection.hpp"
#include "restart.hpp"

using size_to = std::optional<size_t>;

namespace parameters
{

    enum class RecombinationWeights
    {
        DEFAULT,
        EQUAL,
        HALF_POWER_LAMBDA
    };


    struct Modules
    {
        bool elitist = false;
        bool active = false;
        bool orthogonal = false;
        bool sequential_selection = false;
        bool threshold_convergence = false;
        bool sample_sigma = false;
        RecombinationWeights weights = RecombinationWeights::DEFAULT;
        sampling::BaseSampler sampler = sampling::BaseSampler::GAUSSIAN;
        sampling::Mirror mirrored = sampling::Mirror::NONE;
        mutation::StepSizeAdaptation ssa = mutation::StepSizeAdaptation::CSA;
        bounds::CorrectionMethod bound_correction = bounds::CorrectionMethod::NONE;
        restart::StrategyType local_restart = restart::StrategyType::IPOP;
    };

    struct Stats
    {
        size_t t = 0;
        size_t evaluations = 0;
        double target = 1e-8;
        size_t max_generations = -1;
        size_t budget = 100000;
        Vector xopt = Vector(0);
        double fopt = std::numeric_limits<double>::infinity();
    };

    struct Strategy
    {
        size_t lambda;
        size_t mu;

        std::shared_ptr<bounds::BoundCorrection> bounds;
    
        Strategy(const size_t dim, const Modules &mod, const size_to l = std::nullopt, const size_to m = std::nullopt);

    };

    struct Weights
    {
        Vector w;
        Vector p;
        Vector n;

        double mueff, mueff_neg;
        double c1, cmu, cc;

        Weights(const size_t dim, const size_t mu, const size_t lambda, const Modules &m);

        void weights_default(const size_t lambda);

        void weights_equal(const size_t mu);

        void weights_half_power_lambda(const size_t mu, const size_t lambda);

        Vector clipped() const;
    };

    struct Dynamic
    {
        Vector m, m_old, dm;
        Vector pc, ps, d;
        Matrix B, C;
        Matrix inv_root_C;
        double dd;
        double chiN;
        double sigma = .5;
        double s = 0;
        bool hs = true;

        Dynamic(const size_t dim);

        void adapt_evolution_paths(const Weights &w, const std::shared_ptr<mutation::Strategy>& mutation_strategy, const Stats& stats, const Strategy& strat);

        void adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const Strategy &strat);

        void restart();

        bool perform_eigendecomposition(const Stats &stats);
    };

    struct Parameters
    {
        size_t dim;
        Modules mod;
        Dynamic dyn;
        Stats stats;
        Strategy strat;        
        Weights weights;

        std::shared_ptr<sampling::Sampler> sampler;
        std::shared_ptr<mutation::Strategy> mutation_strategy;
        std::shared_ptr<selection::Strategy> selection_strategy;
        std::shared_ptr<restart::Strategy> restart_strategy;

        Population pop;
        Population old_pop;

        Parameters(const size_t dim);

        void adapt();

        void restart();
    };
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);
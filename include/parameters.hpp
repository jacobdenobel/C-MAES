#pragma once

#include "population.hpp"
#include "sampling.hpp"
#include "bounds.hpp"

using size_to = std::optional<size_t>;

namespace parameters
{

    enum class BaseSampler
    {
        GAUSSIAN,
        SOBOL,
        HALTON,
        TESTER
    };

    enum class Mirrored
    {
        NONE,
        MIRRORED,
        PAIRWISE
    };

    enum class StepSizeAdaptation
    {
        CSA,
        TPA,
        MSR,
        XNES,
        MXNES,
        LPXNES,
        PSR
    };

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
        BaseSampler sampler = BaseSampler::GAUSSIAN;
        Mirrored mirrored = Mirrored::NONE;
        StepSizeAdaptation ssa = StepSizeAdaptation::CSA;
        bounds::CorrectionMethod bound_correction = bounds::CorrectionMethod::MIRROR;
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
        double seq_cutoff_factor = 1.;
        size_t seq_cutoff;
        std::shared_ptr<bounds::BoundCorrection> bounds;
        
        double init_threshold = 0.1;
        double decay_factor = 0.995;
        double succes_ratio = .25;
        double a_tpa = 0.5;
        double b_tpa = 0.0;
        double beta;

        Strategy(const size_t dim, const Modules &mod, const size_to l = std::nullopt, const size_to m = std::nullopt);

        [[nodiscard]] double threshold(const Stats &s) const;
    };

    struct Weights
    {
        Vector w;
        Vector p;
        Vector n;

        double mueff, mueff_neg;
        double c1, cmu, cc, cs;
        double damps;

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
        double rank_tpa = 0.0;
        bool hs = true;

        Dynamic(const size_t dim);

        void adapt_evolution_paths(const Weights &w, const Stats &stats, const Strategy &strat);

        void adapt_sigma(const Weights &w, const Modules &m, const Population &pop, const Population &old_pop, const Stats &stats, const Strategy &strat);

        void adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const Strategy &strat);

        void restart();

        void perform_eigendecomposition(const Stats &stats);

        void adapt(const Weights &w, const Stats &stats, const Strategy &strat, const Modules &m, const Population &pop, const Population &old_pop);
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

        Population pop;
        Population old_pop;

        Parameters(const size_t dim);

        void adapt();

        static std::shared_ptr<sampling::Sampler> get_sampler(const size_t dim, const Modules &mod, const Strategy &strat);
    };
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &dt);
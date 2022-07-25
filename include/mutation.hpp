#pragma once

#include "common.hpp"
#include "sampling.hpp"
#include "population.hpp"

namespace parameters {
    struct Stats;
    struct Parameters;   
    struct Weights;
    struct Dynamic;
    struct Strategy;
}

namespace bounds {
    struct BoundCorrection;
}

namespace mutation {
    
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

    class ThresholdConvergence {
        
        double init_threshold = 0.1;
        double decay_factor = 0.995;

    public:
        virtual void scale(Matrix& z, const parameters::Stats& s, const std::shared_ptr<bounds::BoundCorrection>& bounds);
    };

    struct NoThresholdConvergence : ThresholdConvergence {
        void scale(Matrix& z, const parameters::Stats& s, const std::shared_ptr<bounds::BoundCorrection>& bounds) override {}
    };

    class SequentialSelection {
        double seq_cutoff_factor;
        size_t seq_cutoff;

    public:
        SequentialSelection(const sampling::Mirror& m, const size_t mu, const double seq_cutoff_factor = 1.0) :
            seq_cutoff_factor(m == sampling::Mirror::PAIRWISE ? std::max(2., seq_cutoff_factor) : seq_cutoff_factor),
            seq_cutoff(static_cast<size_t>(mu * seq_cutoff_factor)) {

        }
        virtual bool break_conditions(const size_t i, const double f, double fopt, const sampling::Mirror& m);
    };

    struct NoSequentialSelection: SequentialSelection {
        
        using SequentialSelection::SequentialSelection;
        
        bool break_conditions(const size_t i, const double f, double fopt, const sampling::Mirror& m) override { return false; }
    };

    struct SigmaSampler {
        double beta;
        
        SigmaSampler(const double d): beta(std::log(2.0) / std::max((std::sqrt(d) * std::log(d)), 1.0)) {}

        virtual void sample(const double sigma, Population& pop) const {
            pop.s = sampling::Random<std::lognormal_distribution<>>(pop.s.size(),
                std::lognormal_distribution<>(std::log(sigma), beta))();
        }
    };

    struct NoSigmaSampler : SigmaSampler  {
        
        using SigmaSampler::SigmaSampler;

        void sample(const double sigma, Population& pop) const override {
            pop.s = pop.s.Constant(pop.s.size(), sigma);
        }
    };

    struct Strategy {
        std::shared_ptr<ThresholdConvergence> tc;
        std::shared_ptr<SequentialSelection> sq;
        std::shared_ptr<SigmaSampler> ss;
        double cs;

        Strategy(
                const std::shared_ptr<ThresholdConvergence>& threshold_covergence,
                const std::shared_ptr<SequentialSelection>& sequential_selection,
                const std::shared_ptr<SigmaSampler>& sigma_sampler,
                const double cs
            ): 
            tc(threshold_covergence), sq(sequential_selection), ss(sigma_sampler), cs(cs) {}

        virtual void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) = 0;
        
        virtual void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) = 0;

        void adapt(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat);
    };

    struct CSA: Strategy {
        double damps;
        
        CSA(const std::shared_ptr<ThresholdConvergence>& threshold_covergence,
            const std::shared_ptr<SequentialSelection>& sequential_selection,
            const std::shared_ptr<SigmaSampler>& sigma_sampler,
            const double cs, const double damps
        ): Strategy(threshold_covergence, sequential_selection, sigma_sampler, cs), damps(damps) {}
        
        void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) override;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
        
    };

    struct TPA: CSA {
        using CSA::CSA;

        double a_tpa = 0.5;
        double b_tpa = 0.0;
        double rank_tpa = 0.0;

        void mutate(std::function<double(Vector)> objective, const size_t n_offspring, parameters::Parameters& p) override;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    struct MSR: CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    struct PSR : CSA {
        double succes_ratio = .25;

        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    struct XNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    struct MXNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    struct LPXNES : CSA {
        using CSA::CSA;

        void adapt_sigma(const parameters::Weights& w, parameters::Dynamic& dyn, Population& pop,
            const Population& old_pop, const parameters::Stats& stats, const parameters::Strategy& strat) override;
    };

    inline std::shared_ptr<Strategy> get(const StepSizeAdaptation& ssa,  
         const bool threshold_convergence, const bool sequential_selection, const bool sample_sigma,
         const sampling::Mirror& m, const size_t mu, const double mueff, 
         const double d
    ) {
        
        auto& tc = threshold_convergence ? std::make_shared<ThresholdConvergence>()
            : std::make_shared<NoThresholdConvergence>();
       
        auto& sq = sequential_selection ? std::make_shared<SequentialSelection>(m, mu) :
            std::make_shared<NoSequentialSelection>(m, mu);
        
        auto& ss = (sample_sigma or ssa == StepSizeAdaptation::LPXNES) ? 
            std::make_shared<SigmaSampler>(d) 
            : std::make_shared<NoSigmaSampler>(d);
        
        double cs = 0.3;
        switch (ssa)
        {
        case StepSizeAdaptation::TPA:
            return std::make_shared<TPA>(tc, sq, ss, cs, 0.0);
        case StepSizeAdaptation::MSR:
            return std::make_shared<MSR>(tc, sq, ss, cs, 0.0);
        case StepSizeAdaptation::XNES:
            cs = mueff / (2.0 * std::log(std::max(2., d)) * sqrt(d));
            return std::make_shared<XNES>(tc, sq, ss, cs, 0.0);
        case StepSizeAdaptation::MXNES:
            cs = 1.;
            return std::make_shared<MXNES>(tc, sq, ss, cs, 0.0);
        case StepSizeAdaptation::LPXNES:
            cs = 9.0 * mueff / (10.0 * sqrt(d));
            return std::make_shared<LPXNES>(tc, sq, ss, cs, 0.0);
        case StepSizeAdaptation::PSR:
            cs = .9;
            return std::make_shared<PSR>(tc, sq, ss, cs, 0.0);
        default:
        case StepSizeAdaptation::CSA:
            cs = (mueff + 2.0) / (d + mueff + 5.0);
            return std::make_shared<CSA>(tc, sq, ss, cs,
                1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs)
            );
        }     
    }
}
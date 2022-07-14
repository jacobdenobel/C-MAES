#pragma once

#include "common.hpp"

namespace parameters
{

    struct Stats
    {
        size_t t = 0;
        size_t used_budget = 0;
        double target = 1e-8;
        size_t max_generations = 100;
        size_t budget = 1000;
        Vector xopt = Vector(0);
        double fopt = std::numeric_limits<double>::infinity();
    };

    struct Strategy
    {
        size_t lambda;
        size_t mu;

        Strategy(const size_t dim) : lambda(4 + std::floor(3 * log(dim))), mu(lambda / 2)
        {
        }
    };

    struct Modules
    {
        bool elitist = false;
        bool active = false;
    };

    struct Weights
    {
        Vector w;
        Vector p;
        Vector n;

        double mueff, mueff_neg;
        double c1, cmu, cc, cs;
        double damps;

        Weights(const size_t dim, const size_t mu, const size_t lambda) : w(lambda), p(mu), n(lambda - mu)
        {
            const double base = std::log((static_cast<double>(lambda) + 1.) / 2.0);
            const double d = static_cast<double>(dim);

            for (size_t i = 0; i < p.size(); ++i)
                p(i) = base - std::log(static_cast<double>(i + 1));

            for (size_t i = 0; i < n.size(); ++i)
                n(i) = base - std::log(static_cast<double>(i + 1 + p.size()));

            mueff = std::pow(p.sum(), 2) / p.dot(p);
            mueff_neg = std::pow(n.sum(), 2) / n.dot(n);
            p /= p.sum();

            c1 = 2.0 / (pow(d + 1.3, 2) + mueff);
            cmu = std::min(1.0 - c1, 2.0 * ((mueff - 2.0 + (1.0 / mueff)) / (pow(d + 2.0, 2) + (2.0 * mueff / 2))));
            cc = (4.0 + (mueff / d)) / (d + 4.0 + (2.0 * mueff / d));
            cs = (mueff + 2.0) / (d + mueff + 5.0);
            damps = 1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs);

            const double amu_neg = 1.0 + (c1 / static_cast<double>(mu));
            const double amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
            const double aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);

            const double neg_scaler = std::min(amu_neg, std::min(amueff_neg, aposdef_neg));
            n *= neg_scaler / n.cwiseAbs().sum();
            w << p, n;
        }
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

        Dynamic(const size_t dim) : m(Vector::Ones(dim) * .5), m_old(dim), dm(dim), pc(dim), ps(dim), d(Vector::Ones(dim)),
                                    B(Matrix::Identity(dim, dim)), C(Matrix::Identity(dim, dim)),
                                    inv_root_C(Matrix::Identity(dim, dim)), dd(static_cast<double>(dim)),
                                    chiN(sqrt(dd) * (1.0 - 1.0 / (4.0 * dd) + 1.0 / (21.0 * pow(dd, 2.0))))
        {
        }

        void adapt_evolution_paths(const Weights &w, const Stats &stats, const Strategy &strat)
        {
            dm = (m - m_old) / sigma;
            ps = (1.0 - w.cs) * ps + (sqrt(w.cs * (2.0 - w.cs) * w.mueff) * inv_root_C * dm);

            const double actual_ps_length = ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * (stats.used_budget / strat.lambda)));
            const double expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * chiN;

            hs = actual_ps_length < expected_ps_length;
            pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
        }

        void adapt_sigma(const Weights &w)
        {
            sigma *= std::exp((w.cs / w.damps) * ((ps.norm() / chiN) - 1));
        }

        void adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const Strategy &strat)
        {
            const auto rank_one = w.c1 * pc * pc.transpose();
            const auto dhs = (1 - hs) * w.cc * (2.0 - w.cc);
            const auto old_C = (1 - (w.c1 * dhs) - w.c1 - (w.cmu * w.p.sum())) * C;

            Matrix rank_mu;
            if (m.active)
            {
                auto weights = w.w.topRows(pop.Y.cols());
                auto neg_scaler = dd / (inv_root_C * pop.Y).colwise().norm().array().pow(2).transpose();
                auto w2 = (weights.array() < 0).select(weights.array() * neg_scaler, weights);

                rank_mu = w.cmu * ((pop.Y.array().rowwise() * w2.array().transpose()).matrix() * pop.Y.transpose());
            }
            else
            {
                rank_mu = w.cmu * ((pop.Y.leftCols(strat.mu).array().rowwise() * w.p.array().transpose()).matrix() * pop.Y.leftCols(strat.mu).transpose());
            }
            C = old_C + rank_one + rank_mu;

            C = C.triangularView<Eigen::Upper>().toDenseMatrix() +
                C.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().transpose();
        }
        void perform_eigendecomposition()
        {
            Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(C);
            if (eigensolver.info() != Eigen::Success)
                abort();
            d = eigensolver.eigenvalues().cwiseSqrt();
            B = eigensolver.eigenvectors();

            inv_root_C = (B * d.cwiseInverse().asDiagonal()) * B.transpose();
        }

        void adapt(const Weights &w, const Stats &stats, const Strategy &strat, const Modules &m, const Population &pop)
        {
            adapt_evolution_paths(w, stats, strat);
            adapt_sigma(w);
            adapt_covariance_matrix(w, m, pop, strat);
            perform_eigendecomposition();
        }
    };

    struct Parameters
    {
        size_t dim;
        Dynamic dyn;
        Stats stats;
        Strategy strat;
        Modules mod;
        Weights weights;

        Population pop;
        Population old_pop;

        Parameters(const size_t dim) : dim(dim), dyn(dim), strat(dim),
                                       weights(dim, strat.mu, strat.lambda),
                                       pop(dim, strat.lambda), old_pop(dim, strat.lambda)
        {
        }

        void adapt()
        {
            dyn.adapt(weights, stats, strat, mod, pop);
            old_pop = pop;
            stats.t++;
        }
    };
}

std::ostream &operator<<(std::ostream &os, const parameters::Stats &s)
{
    return os
           << "Stats"
           << " t=" << s.t 
           << " evals=" << s.used_budget 
           << " xopt=("
           << s.xopt.transpose()
           << ") fopt="
           << s.fopt;
}

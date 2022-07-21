#include "parameters.hpp"

//! Assumes the vector to be arready sorted
double median(const Vector &x)
{
    if (x.size() % 2 == 0)
        return (x(x.size() / 2) + x(x.size() / 2 - 1)) / 2.0;
    return x(x.size() / 2);
}

Vector searchsorted(const Vector& query, const Vector& database){
    Vector res(query.size());
    auto i = 0;
    for(const auto& xi: query){
        auto it = std::find(std::begin(database), std::end(database), xi);
        res(++i) = std::distance(std::begin(database), it);
    }
    return res;
}



namespace parameters
{
    Dynamic::Dynamic(const size_t dim) : m(Vector::Random(dim) * 5), m_old(dim), dm(Vector::Zero(dim)), pc(Vector::Zero(dim)), ps(Vector::Zero(dim)), d(Vector::Ones(dim)),
                                         B(Matrix::Identity(dim, dim)), C(Matrix::Identity(dim, dim)),
                                         inv_root_C(Matrix::Identity(dim, dim)), dd(static_cast<double>(dim)),
                                         chiN(sqrt(dd) * (1.0 - 1.0 / (4.0 * dd) + 1.0 / (21.0 * pow(dd, 2.0))))
    {
    }

    void Dynamic::adapt_evolution_paths(const Weights &w, const Stats &stats, const Strategy &strat)
    {
        dm = (m - m_old) / sigma;
        ps = (1.0 - w.cs) * ps + (sqrt(w.cs * (2.0 - w.cs) * w.mueff) * inv_root_C * dm);

        const double actual_ps_length = ps.norm() / sqrt(1.0 - pow(1.0 - w.cs, 2.0 * (stats.evaluations / strat.lambda)));
        const double expected_ps_length = (1.4 + (2.0 / (dd + 1.0))) * chiN;

        hs = actual_ps_length < expected_ps_length;
        pc = (1.0 - w.cc) * pc + (hs * sqrt(w.cc * (2.0 - w.cc) * w.mueff)) * dm;
    }

    void Dynamic::adapt_sigma(const Weights &w, const Modules &m, const Population &pop, const Population &old_pop, const Stats &stats, const Strategy &strat)
    {

        switch (m.ssa)
        {
        case StepSizeAdaptation::CSA:
            sigma *= std::exp((w.cs / w.damps) * ((ps.norm() / chiN) - 1));
            break;
        case StepSizeAdaptation::LPXNES:
        {
            const auto z = std::exp(w.cs * pop.s.array().log().matrix().dot(w.clipped()));
            sigma = std::pow(sigma, 1.0 - w.cs) * z;
        }
        break;
        case StepSizeAdaptation::XNES:
        {
            const double z = ((inv_root_C * pop.Y).colwise().norm().array().pow(2.) - dd).matrix() * w.clipped();
            sigma *= std::exp((w.cs / std::sqrt(dd)) * z);
        }
        break;
        case StepSizeAdaptation::TPA:
            s = ((1.0 - w.cs) * s) + (w.cs * rank_tpa);
            sigma *= std::exp(s);
            break;

        case StepSizeAdaptation::MSR:
        {
            if (stats.t != 0)
            {
                //TODO: check why this sucks on sphere 1.8k evals 5d
                const double lambda = static_cast<double>(strat.lambda);
                const double k = (pop.f.array() < median(old_pop.f)).cast<double>().sum();
                const auto z = (2.0 / lambda) * (k - ((lambda + 1.0) / 2.0));
                s = ((1.0 - w.cs) * s) + (w.cs * z);
                sigma *= std::exp(s / (2.0 - (2.0 / dd)));
            }
        }
        break;
        case StepSizeAdaptation::MXNES:
            if (stats.t != 0)
            {
                const auto z = (w.mueff * std::pow((inv_root_C * dm).norm(), 2)) - dd;
                sigma *= std::exp((w.cs / dd) * z);
            }
            break;
        case StepSizeAdaptation::PSR:
            if (stats.t != 0)
            {
                const auto n = std::min(pop.n, old_pop.n);
                auto combined = Vector(n+n);
                combined << pop.f.head(n), old_pop.f.head(n);
                const auto idx = utils::sort_indexes(combined);
                combined = combined(idx).eval();

                auto r = searchsorted(pop.f.head(n), combined);
                auto r_old = searchsorted(old_pop.f.head(n), combined);

                const auto z = (r_old - r).sum() / std::pow(n, 2) - strat.succes_ratio;

                s = (1.0 - w.cs) * s + (w.cs * z);
                sigma *= std::exp(s / (2.0 - (2.0 / dd)));
            }
            break; 
        }
    }

    void Dynamic::adapt_covariance_matrix(const Weights &w, const Modules &m, const Population &pop, const Strategy &strat)
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

    void Dynamic::restart()
    {
        auto dim = C.cols();
        B = Matrix::Identity(dim, dim);
        C = Matrix::Identity(dim, dim);
        inv_root_C = Matrix::Identity(dim, dim);
        d.setOnes();
        m = Vector::Random(dim) * 5;
        m_old.setZero();
        pc.setZero();
        ps.setZero();
        sigma = .5;
        s = 0;
    }

    void Dynamic::perform_eigendecomposition(const Stats &stats)
    {
        Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(C);
        if (eigensolver.info() != Eigen::Success)
        {
            // TODO: check why this sometimes happens on the first eval (sphere 60d)
            std::cout << "eigensolver failed, we need to restart t(" << stats.t << ")\n";
            return restart();
        }

        d = eigensolver.eigenvalues().cwiseSqrt();
        B = eigensolver.eigenvectors();

        inv_root_C = (B * d.cwiseInverse().asDiagonal()) * B.transpose();
    }

    void Dynamic::adapt(const Weights &w, const Stats &stats, const Strategy &strat, const Modules &m, const Population &pop, const Population &old_pop)
    {
        adapt_evolution_paths(w, stats, strat);
        adapt_sigma(w, m, pop, old_pop, stats, strat);
        adapt_covariance_matrix(w, m, pop, strat);
        perform_eigendecomposition(stats);
    }
}
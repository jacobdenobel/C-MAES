#include "parameters.hpp"

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
    
    void Dynamic::adapt_sigma(const Weights &w, const Modules& m) 
    {
        switch (m.ssa)
        {
        case StepSizeAdaptation::CSA:
            sigma *= std::exp((w.cs / w.damps) * ((ps.norm() / chiN) - 1));
            break;
        case StepSizeAdaptation::LPXNES:
            break;

        // need old population
        case StepSizeAdaptation::XNES:
            break;
        case StepSizeAdaptation::MSR:
            break;
         case StepSizeAdaptation::TPA:
            break;   
        case StepSizeAdaptation::MXNES:
            break;    
        case StepSizeAdaptation::PSR:
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

    void Dynamic::adapt(const Weights &w, const Stats &stats, const Strategy &strat, const Modules &m, const Population &pop)
    {
        adapt_evolution_paths(w, stats, strat);
        adapt_sigma(w, m);
        adapt_covariance_matrix(w, m, pop, strat);
        perform_eigendecomposition(stats);
    }
}
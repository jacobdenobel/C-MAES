#include "parameters.hpp"

namespace parameters
{

    Weights::Weights(const size_t dim, const size_t mu, const size_t lambda, const Modules &m)
        : w(lambda), p(mu), n(lambda - mu)
    {
        const double d = static_cast<double>(dim);

        switch (m.weights)
        {
        case RecombinationWeights::EQUAL:
            weights_equal(mu);
            break;
        case RecombinationWeights::HALF_POWER_LAMBDA:
            weights_half_power_lambda(mu, lambda);
            break;
        case RecombinationWeights::DEFAULT:
            weights_default(lambda);
            break;
        }

        mueff = std::pow(p.sum(), 2) / p.dot(p);
        mueff_neg = std::pow(n.sum(), 2) / n.dot(n);
        p /= p.sum();

        c1 = 2.0 / (pow(d + 1.3, 2) + mueff);
        cmu = std::min(1.0 - c1, 2.0 * ((mueff - 2.0 + (1.0 / mueff)) / (pow(d + 2.0, 2) + (2.0 * mueff / 2))));
        cc = (4.0 + (mueff / d)) / (d + 4.0 + (2.0 * mueff / d));
        

        switch (m.ssa)
        {
        case StepSizeAdaptation::CSA:
            cs = (mueff + 2.0) / (d + mueff + 5.0);
            break;
        case StepSizeAdaptation::XNES:
            cs = mueff / (2.0 * std::log(std::max(2., d)) * sqrt(d));
            break;
        case StepSizeAdaptation::MXNES:
            cs = 1.;
            break;
        case StepSizeAdaptation::LPXNES:
            cs = 9.0 * mueff / (10.0 * sqrt(d));
            break;
        case StepSizeAdaptation::PSR:
            cs = .9;
            break;
        default:
            cs = .3;
        }
        damps = 1.0 + (2.0 * std::max(0.0, sqrt((mueff - 1.0) / (d + 1)) - 1) + cs);
        const double amu_neg = 1.0 + (c1 / static_cast<double>(mu));
        const double amueff_neg = 1.0 + ((2.0 * mueff_neg) / (mueff + 2.0));
        const double aposdef_neg = (1.0 - c1 - cmu) / (d * cmu);

        const double neg_scaler = std::min(amu_neg, std::min(amueff_neg, aposdef_neg));
        n *= neg_scaler / n.cwiseAbs().sum();
        w << p, n;
    }

    void Weights::weights_default(const size_t lambda)
    {
        const double base = std::log((static_cast<double>(lambda) + 1.) / 2.0);
        for (auto i = 0; i < p.size(); ++i)
            p(i) = base - std::log(static_cast<double>(i + 1));

        for (auto i = 0; i < n.size(); ++i)
            n(i) = base - std::log(static_cast<double>(i + 1 + p.size()));
    }

    void Weights::weights_equal(const size_t mu)
    {
        const double wi = 1. / static_cast<double>(mu);
        p.setConstant(wi);
        n.setConstant(-wi);
    }

    void Weights::weights_half_power_lambda(const size_t mu, const size_t lambda)
    {
        const double dmu = static_cast<double>(mu);
        const double base = (1.0 / pow(2.0, dmu)) / dmu;
        const double delta = static_cast<double>(lambda - mu);
        const double base2 = (1.0 / pow(2.0, delta)) / delta;

        for (auto i = 0; i < p.size(); ++i)
            p(i) = dmu / pow(2.0, static_cast<double>(i + 1)) + base;

        for (auto i = 0; i < n.size(); ++i)
            n(n.size() - i) = 1.0 / pow(2.0, static_cast<double>(i + 1)) + base2;
    }


    Vector Weights::clipped() const {
        return (w.array() > 0).select(w, Vector::Zero(w.size()));
    }

}
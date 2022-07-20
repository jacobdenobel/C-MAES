#include "c++maes.hpp"

void ModularCMAES::mutate(std::function<double(Vector)> objective)
{
    for (size_t i = 0; i < p.pop.Z.cols(); ++i)
        p.pop.Z.col(i) = (*p.sampler)();

    if (p.mod.threshold_convergence)
        scale_with_threshold(p.pop.Z, p.strat.threshold(p.stats));
    
    // if (p.mod.ssa == parameters::StepSizeAdaptation::LPXNES or p.mod.sample_sigma){
    //     // auto sampler = sampling::Random(p.dim, std::lognormal_distribution<>(std::log(p.dyn.sigma), p.strat.beta));
    //
    // }else{
    //     p.pop.s = p.pop.s.Constant(p.pop.s.size(), p.dyn.sigma).eval();
    // }
    
    
    p.pop.Y = p.dyn.B * (p.dyn.d.asDiagonal() * p.pop.Z);
    p.pop.X = (p.dyn.sigma * p.pop.Y).colwise() + p.dyn.m;

    for (size_t i = 0; i < p.pop.X.cols(); ++i)
    {
        p.pop.f(i) = objective(p.pop.X.col(i));
        p.stats.evaluations++;
        if (sequential_break_conditions(i, p.pop.f(i)))
            break;
    }
}

void ModularCMAES::select()
{
    if (p.mod.mirrored == parameters::Mirrored::PAIRWISE)
    {
        assert(p.pop.f.size() % 2 == 0);
        std::vector<int> idx;
        for (size_t i = 0, j = 0; i < p.pop.f.size(); i += 2)
            idx[++j] = i + (1 * (p.pop.f(i) > p.pop.f(i + 1)));
    }

    if (p.mod.elitist and p.stats.t != 0)
        p.pop += p.old_pop;

    p.pop.sort();
    p.pop.resize_cols(p.strat.lambda);

    if (p.pop.f(0) < p.stats.fopt)
    {
        p.stats.fopt = p.pop.f(0);
        p.stats.xopt = p.pop.X(Eigen::all, 0);
    }
}

void ModularCMAES::recombine()
{
    p.dyn.m_old = p.dyn.m;
    p.dyn.m = p.dyn.m_old + ((p.pop.X.leftCols(p.strat.mu).colwise() - p.dyn.m_old) * p.weights.p);
}

bool ModularCMAES::step(std::function<double(Vector)> objective)
{
    mutate(objective);
    select();
    recombine();
    p.adapt();
    return !break_conditions();
}

void ModularCMAES::operator()(std::function<double(Vector)> objective)
{
    while (step(objective))
    {
        if (p.stats.t % (p.dim * 2) == 0 and verbose)
            std::cout << p.stats << std::endl;
    }
    if (verbose)
        std::cout << p.stats << std::endl;
}

bool ModularCMAES::sequential_break_conditions(const size_t i, const double f) const
{
    if (p.mod.sequential_selection)
        return f < p.stats.fopt and i >= p.strat.seq_cutoff and (p.mod.mirrored != parameters::Mirrored::PAIRWISE or i % 2 == 0);
    return false;
}

bool ModularCMAES::break_conditions() const
{
    const bool target_reached = p.stats.target >= p.stats.fopt;
    const bool budget_used_up = p.stats.evaluations >= p.stats.budget;
    const bool exceed_gens = p.stats.t >= p.stats.max_generations;

    return exceed_gens or target_reached or budget_used_up;
}

void scale_with_threshold(Matrix& z, const double t){
    const auto norm = z.colwise().norm().array().replicate(z.cols() - 1, 1);
    z = (norm < t).select(z.array() * ((t + (t - norm)) / norm), z);
}
#include "c++maes.hpp"


void ModularCMAES::recombine()
{
	p.dyn.m_old = p.dyn.m;
	p.dyn.m = p.dyn.m_old + ((p.pop.X.leftCols(p.strat.mu).colwise() - p.dyn.m_old) * p.weights.p);
}

bool ModularCMAES::step(std::function<double(Vector)> objective)
{
	p.mutation_strategy->mutate(objective, p.pop.Z.cols(), p);
	p.selection_strategy->select(p);
	recombine();
	p.adapt();
	return !break_conditions();
}

void ModularCMAES::operator()(std::function<double(Vector)> objective)
{
	while (step(objective))
	{
		if (p.stats.t % (p.dim * 2) == 0 and verbose)
			std::cout << p.stats << " sigma: " << p.dyn.sigma << std::endl;
	}
	if (verbose)
		std::cout << p.stats << std::endl;
}

bool ModularCMAES::break_conditions() const
{
	const auto target_reached = p.stats.target >= p.stats.fopt;
	const auto budget_used_up = p.stats.evaluations >= p.stats.budget;
	const auto exceed_gens = p.stats.t >= p.stats.max_generations;

	return exceed_gens or target_reached or budget_used_up;
}


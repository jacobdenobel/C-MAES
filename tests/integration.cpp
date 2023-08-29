

#include "../include/c_maes.hpp"
#include "ioh.hpp"

#include <chrono>


struct ioh_function {
    std::shared_ptr<ioh::problem::RealSingleObjective> p;
    std::vector<double> v;
    ioh_function(const std::shared_ptr<ioh::problem::RealSingleObjective>& p): p(p) {
        v.resize(p->meta_data().n_variables);
    }

    double operator()(const Vector &x) {
        Eigen::VectorXd::Map(&v[0], x.size()) = x;
        return (*p)(v);
    }
};



int run_ioh(const int d, const parameters::Modules& m)
{
    const size_t reps = 50;
    std::cout << d << std::endl;
    auto& factory = ioh::problem::ProblemRegistry<ioh::problem::RealSingleObjective>::instance();
    auto logger = ioh::logger::Analyzer();
    
    ioh::suite::BBOB suite{ {1}, {1}, {d} };
    
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (const auto& problem: suite){
        problem->attach_logger(logger);

        std::cout << *problem << std::endl;

        for (size_t i = 0; i < reps; ++i){
            parameters::Parameters p(d, m);
            p.stats.target = problem->optimum().y + 1e-8;
            p.stats.budget = static_cast<size_t>(1e4 * d);
            p.verbose = false;
            ModularCMAES cma(p);
         
            cma(ioh_function(problem));
            // std::cout << (*problem).state() << std::endl;
            problem->reset();
        }
    }
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "Time elapsed: " << duration.count() / 1000.0 << std::endl;
    return 0;
}


int main(int argc, char *argv[]){
    const int f = argc < 2 ? 7 : std::stoi(argv[1]);
    const int d = argc < 3 ? 5 : std::stoi(argv[2]);
    const int s = argc < 4 ? 10 : std::stoi(argv[3]);
    const bool v = static_cast<bool>(argc < 5 ? 0 : std::stoi(argv[4]));
    

    parameters::Modules m;

    // m.elitist = false;
    // m.active = false;
    // m.orthogonal = false;
    // m.sequential_selection = false;
    // m.threshold_convergence = false;
    // m.sample_sigma = false;
    // m.weights = parameters::RecombinationWeights::DEFAULT;
    // m.sampler = sampling::BaseSampler::HALTON;
    // m.mirrored = sampling::Mirror::NONE;
    // m.ssa = mutation::StepSizeAdaptation::CSA;
    // m.bound_correction = bounds::CorrectionMethod::NONE;
    // m.restart_strategy = restart::StrategyType::NONE;


    rng::set_seed(s);
    if (f == 0)
        return run_ioh(d, m);

    auto& factory = ioh::problem::ProblemRegistry<ioh::problem::RealSingleObjective>::instance();
    auto problem = factory.create(f, 1, d);
    std::cout << (*problem) << std::endl;

    parameters::Parameters p(d, m);
    p.stats.target = problem->optimum().y + 1e-8;
    p.stats.budget = static_cast<size_t>(1e4 * d);
    p.verbose = v;

    ModularCMAES cma(p);

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    
    cma(ioh_function(problem));
    
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "Time elapsed: " << duration.count() / 1000.0 << std::endl;
    std::cout << "Time elapsed (per iter): " << (static_cast<double>(duration.count()) / cma.p.stats.t) / 1000.0 << std::endl;
    std::cout << problem->state() << std::endl;
}    
    

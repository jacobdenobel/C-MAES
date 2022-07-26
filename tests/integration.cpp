

#include "c++maes.hpp"
#include "ioh.hpp"

#include <chrono>


double sphere(const Vector &x)
{
    double res = 0;
    for (auto &xi : x)
        res += xi * xi;
    return res;
}


std::pair<double, size_t> compute_ert(const std::vector<size_t> &running_times, const size_t budget)
{
    size_t successfull_runs = 0, total_rt = 0;

    for (const auto &rt : running_times)
    {
        if (rt < budget)
            successfull_runs++;
        total_rt += rt;
    }
    return {static_cast<double>(total_rt) / successfull_runs, successfull_runs};
}


struct ioh_function {
    std::shared_ptr<ioh::problem::Real> p;
    std::vector<double> v;
    ioh_function(const std::shared_ptr<ioh::problem::Real>& p): p(p) {
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
    auto& factory = ioh::problem::ProblemRegistry<ioh::problem::Real>::instance();
    auto logger = ioh::logger::Analyzer();
    
    ioh::suite::BBOB suite{ {1}, {1}, {d} };
    
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (const auto& problem: suite){
        problem->attach_logger(logger);

        std::cout << *problem << std::endl;

        for (size_t i = 0; i < reps; ++i){
            parameters::Parameters p(d, m);
            p.stats.target = problem->objective().y + 1e-8;
            p.stats.budget = static_cast<size_t>(1e4 * d);
            p.verbose = false;
            ModularCMAES cma(p);
         
            cma(ioh_function(problem));
            std::cout << (*problem).state() << std::endl;
            problem->reset();
        }
    }
    auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "Time elapsed: " << duration.count() / 1000.0 << std::endl;
    return 0;
}


int main(int argc, char *argv[]){
    const int f = argc < 2 ? 5 : std::stoi(argv[1]);
    const int d = argc < 3 ? 5 : std::stoi(argv[2]);
    const int s = argc < 4 ? 1 : std::stoi(argv[3]);
    const bool v = static_cast<bool>(argc < 5 ? 0 : std::stoi(argv[4]));
    

    parameters::Modules m;

    m.elitist = false;
    m.active = false;
    m.orthogonal = false;
    m.sequential_selection = false;
    m.threshold_convergence = false;
    m.sample_sigma = false;
    m.weights = parameters::RecombinationWeights::DEFAULT;
    m.sampler = sampling::BaseSampler::GAUSSIAN;
    m.mirrored = sampling::Mirror::NONE;
    m.ssa = mutation::StepSizeAdaptation::CSA;
    m.bound_correction = bounds::CorrectionMethod::NONE;
    m.restart_strategy = restart::StrategyType::NONE;


    rng::set_seed(s);
    if (f == 0)
        return run_ioh(d, m);

    auto& factory = ioh::problem::ProblemRegistry<ioh::problem::Real>::instance();
    auto problem = factory.create(f, 1, d);
    std::cout << (*problem) << std::endl;

    parameters::Parameters p(d, m);
    p.stats.target = problem->objective().y + 1e-8;
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
    



#include "c++maes.hpp"

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



int main()
{
    const int d = 5;

    sampling::Tester s(d);

    parameters::Parameters p(d);

    ModularCMAES cma(p);

    cma(&sphere);
}

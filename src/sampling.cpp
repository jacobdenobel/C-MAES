#include "sampling.hpp"

namespace sampling
{
    [[nodiscard]] Vector Tester::operator()()
    {
        Vector x(d);
        ++i;
        x.array() = static_cast<double>(i);
        return x;
    };


    [[nodiscard]] Vector Mirrored::operator()()
    {
        if (!mirror)
        {
            previous = (*sampler)();
            mirror = true;
            return previous;
        }
        mirror = false;
        return -previous;
    }

    [[nodiscard]] Vector Orthogonal::operator()()
    {
        if (current >= n)
            current = 0;

        if (!current)
        {
            for (size_t i = 0; i < n; ++i)
                samples.col(i) = (*sampler)();

            auto norm = samples.colwise().norm().asDiagonal();

            qr.compute(samples.transpose());
            samples = ((qr.householderQ() * I).transpose() * norm);
        }
        return samples.col(current++);
    }

    Halton::Halton(const size_t d, const size_t i) : Sampler(d), i(i)
    {
        primes = sieve(std::max(6, static_cast<int>(d)));
        while (primes.size() < d)
            primes = sieve(primes.size() * primes.size());
        primes.resize(d);
    }

    [[nodiscard]] Vector Halton::operator()()
    {
        Vector res(d);
        for (size_t j = 0; j < d; ++j)
            res(j) = ppf(next(i, primes[j]));
        i++;
        return res;
    }

    double Halton::next(int index, int base)
    {
        double y = 1., x = 0.;
        while (index > 0)
        {
            auto dm = divmod(index, base);
            index = dm.first;
            y *= static_cast<double>(base);
            x += static_cast<double>(dm.second) / y;
        }
        return x;
    }

    std::pair<int, int> Halton::divmod(const double top, const double bottom)
    {
        const auto div = static_cast<int>(top / bottom);
        return {div, top - div * bottom};
    }

    std::vector<int> Halton::sieve(const size_t n)
    {
        std::vector<unsigned char> mask(n / 3 + (n % 6 == 2), 1);
        size_t s = static_cast<size_t>(pow(n, .5)) / 3 + 1;
        for (size_t i = 1; i < s; ++i)
        {
            if (mask[i])
            {
                auto k = 3 * i + 1 | 1;
                for (int j = k * k / 3; j < 2 * k - 1; ++j)
                    mask[j] = 0;
                for (int j = k * (k - 2 * (i & 1) + 4) / 3; j < 2 * k - 1; ++j)
                    mask[j] = 0;
            }
        }

        std::vector<int> primes = {2, 3};
        for (size_t i = 1; i < mask.size(); ++i)
            if (mask[i])
                primes.push_back((3 * i + 1) | 1);

        return primes;
    }

    [[nodiscard]] Vector Sobol::operator()()
    {
        Vector res(d);
        i8_sobol(d, &seed, res.data());
        for (size_t j = 0; j < d; ++j)
            res(j) = ppf(res(j));

        seed += d;
        return res;
    }

}
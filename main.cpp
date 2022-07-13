#include <iostream>

#include "c++maes.hpp"

double sphere(const std::vector<double>& x){
    double res = 0;
    for(auto& xi :x)
        res += xi*xi;
    return res;
}
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& x){
    for(auto& xi: x)
        os << xi << ' ';
    return os;
}

int main(){
    auto sampler = std::make_shared<sampling::Tester>(2);
    // sampling::Mirrored mirrored(sampler);
    // sampling::Orthogonal orth(sampler, 2);
    sampling::Halton h(5);
    std::cout << h() << std::endl;
}

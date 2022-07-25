#include <chrono>
#include <iostream>
#include <utility>
#include "Eigen/Dense"

using Vector = Eigen::VectorXd;

int K = 10000000;
int N = 100;
Vector time(std::function<Vector(void)> f) {
	using namespace std::chrono;

	auto start = high_resolution_clock::now();
	auto res = f();
	auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - start);
	std::cout << "Time elapsed: " << duration.count() / 1000.0 << std::endl;
	return res;
}

Vector assign() {
	Vector f(N);

	for (int i = 0; i < K; i++) 
		for (auto& fi : f)
			fi = i;

	return f;
}

Vector constant() {
	Vector f(N);

	for (int i = 0; i < K; i++)
		f = Vector::Constant(N, i);
	return f;
}

int main() {

	auto x = time(assign);
	std::cout << x.transpose() << std::endl;
	auto y = time(constant);
	std::cout << y.transpose() << std::endl;

	 x = time(assign);
	std::cout << x.transpose() << std::endl;
	 y = time(constant);
	std::cout << y.transpose() << std::endl;
}
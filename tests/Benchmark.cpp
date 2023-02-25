#include <omp.h>
#include <iostream>
#include <functional>

#include "BlockMultiplication.h"
#include "StrassenMultiplication.h"

template<typename T>
using multiplication_function = std::function<std::vector<T>(const std::vector<T>&, const std::vector<T>&, size_t)>;

template<typename T>
void TestMatrixAlg(multiplication_function<T> mult_function,
    const std::vector<T>& a, const std::vector<T>& b)
{
    if (a.size() != b.size())
        throw std::exception("Wrong size vectors.");

    double start, end, duration;
    
	omp_set_num_threads(4);
    start = omp_get_wtime();
    mult_function(a, b, a.size());
    end = omp_get_wtime();

    duration = end - start;

    std::cout << "Time: " << duration << "s | ";
	std::cout << std::endl << std::endl;
}

int main() {
	std::random_device rd;
	std::mt19937 eng{ rd() };
	std::uniform_int_distribution<int> int_dist{ 1, 10000 };
	std::uniform_real_distribution<double> double_dist{ 1.0, 10000.0 };

	auto int_gen = [&int_dist, &eng]() -> int {
		return int_dist(eng);
	};

	auto double_gen = [&double_dist, &eng]() -> double {
		return double_dist(eng);
	};

	size_t size = 1024;
	std::vector<int> ai(size * size);
	std::vector<int> bi(size * size);
	std::vector<double> ad(size * size);
	std::vector<double> bd(size * size);

	std::generate(ai.begin(), ai.end(), int_gen);
	std::generate(bi.begin(), bi.end(), int_gen);
	std::generate(ad.begin(), ad.end(), double_gen);
	std::generate(bd.begin(), bd.end(), double_gen);

	// int matrix test
	multiplication_function<int> int_mult;
	std::pair<const char*, multiplication_function<int>> int_tests[] = {
		{"[Fox]", parallelBlockMatrixMultiplication<int>},
		{"[Naive]", parallelNaiveMatrixMultiplication<int>},
		{"[Strassen]", strassenMultiplication_omp<int>},
	};

	// double matrix test
	multiplication_function<double> double_mult;
	std::pair<const char*, multiplication_function<double>> double_tests[] = {
		{"[Fox]", parallelBlockMatrixMultiplication<double>},
		{"[Naive]", parallelNaiveMatrixMultiplication<double>},
	    {"[Strassen]", strassenMultiplication_omp<double>},
	};

	try {
		std::cout << "INT tests:" << std::endl << std::endl;
		for (auto& int_test : int_tests) {
			std::cout << int_test.first << std::endl;
			int_mult = int_test.second;
			TestMatrixAlg(int_mult, ai, bi);
		}

		std::cout << "DOUBLE tests:" << std::endl << std::endl;
		for (auto& double_test : double_tests) {
			std::cout << double_test.first << std::endl;
			double_mult = double_test.second;
			TestMatrixAlg(double_mult, ad, bd);
		}
	}
	catch (std::exception& exception) {
		std::cout << "Error! " << exception.what() << std::endl;
	}
	catch (...) {
		std::cout << "Unknown error." << std::endl;
	}

	return 0;
}
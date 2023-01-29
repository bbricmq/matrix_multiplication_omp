#include <omp.h>
#include <iostream>

#include "fox.h"
#include "strassen.h"

int main() {

    size_t size = 1024;
    std::vector<double> A = createRandomMatrix(size * size);
    std::vector<double> B = createRandomMatrix(size * size);
    omp_set_num_threads(4);
    double t1 = omp_get_wtime();
    std::vector<double> C = parallelBlockMatrixMultiplication(A, B, size * size);
    double t2 = omp_get_wtime();
    std::cout << "Omp Fox time 4 threads: " << t2 - t1 << std::endl;

    omp_set_num_threads(1);
    double t3 = omp_get_wtime();
    std::vector<double> C_block = parallelBlockMatrixMultiplication(A, B, size * size);
    double t4 = omp_get_wtime();
    std::cout << "Seq Fox time: " << t4 - t3 << std::endl;

    double t5 = omp_get_wtime();
    std::vector<double> C_naive = sequentialMatrixMultiplication(A, B, size * size);
    double t6 = omp_get_wtime();
    std::cout << "Naive time: " << t6 - t5 << std::endl;

    double t7 = omp_get_wtime();
    std::vector<double> C_naivee = strassenMultiplication_omp(A, B, size);
    double t8 = omp_get_wtime();
    std::cout << "Strassen time: " << t8 - t7 << std::endl;

    return 0;
}
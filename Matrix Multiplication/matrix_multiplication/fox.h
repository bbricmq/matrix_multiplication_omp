#pragma once
#include <omp.h>
#include <vector>
#include <string>
#include <random>
#include <iostream>
using namespace std;

typedef std::vector<double> Matrix;

bool isEqual(double x, double y);
bool isEqualMatrix(const Matrix& A, const Matrix& B);
bool isSizeCorrect(size_t size, size_t t_count);

Matrix createRandomMatrix(size_t size);
Matrix sequentialMatrixMultiplication(const std::vector<double>& A,
                                      const std::vector<double>& B,
                                      size_t Size);
Matrix sequentialBlockMatrixMultiplication(const std::vector<double>& A,
                                           const std::vector<double>& B,
                                           size_t Size);
Matrix parallelBlockMatrixMultiplication(const std::vector<double>& A,
                                         const std::vector<double>& B,
                                         size_t Size);
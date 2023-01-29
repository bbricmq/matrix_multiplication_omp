#pragma once
#include <omp.h>

#include <vector>
#include <string>
#include <random>
#include <iostream>

bool isEqual(double x, double y);
bool isEqualMatrix(const std::vector<double>& A, const std::vector<double>& B);
bool isSizeCorrect(size_t size, size_t t_count);

std::vector<double> createRandomMatrix(size_t size);
std::vector<double> sequentialMatrixMultiplication(const std::vector<double>& A,
    const std::vector<double>& B,
    size_t Size);
std::vector<double> sequentialBlockMatrixMultiplication(const std::vector<double>& A,
    const std::vector<double>& B,
    size_t Size);
std::vector<double> parallelBlockMatrixMultiplication(const std::vector<double>& A,
    const std::vector<double>& B,
    size_t Size);
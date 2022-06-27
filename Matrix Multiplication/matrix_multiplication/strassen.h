#pragma once
#include <omp.h>
#include <math.h>

#include <algorithm>
#include <iostream>
#include <vector>

std::vector<double> getRandomMatrix(int sz);
std::vector<double> regularMultiplication(const std::vector<double> &a,
                                          const std::vector<double> &b,
                                          unsigned int sz);
std::vector<double> strassenMultiplication_omp(const std::vector<double> &a,
                                               const std::vector<double> &b,
                                               unsigned int sz);
std::vector<double> strassenMultiplication_NoCastToPowerOfTwo_omp(
    const std::vector<double> &a, const std::vector<double> &b,
    unsigned int sz);
std::vector<double> strassenMultiplication(const std::vector<double> &a,
                                           const std::vector<double> &b,
                                           unsigned int sz);
std::vector<double> strassenMultiplication_NoCastToPowerOfTwo(
    const std::vector<double> &a, const std::vector<double> &b,
    unsigned int sz);
std::vector<double> sum(const std::vector<double> &a,
                        const std::vector<double> &b, unsigned int sz);
std::vector<double> diff(const std::vector<double> &a,
                         const std::vector<double> &b, unsigned int sz);
void splitMatrix(const std::vector<double> &mtx, std::vector<double> *a,
                 std::vector<double> *b, std::vector<double> *c,
                 std::vector<double> *d, unsigned int sz);
void mergeMatrix(std::vector<double> *mtx, const std::vector<double> &a,
                 const std::vector<double> &b, const std::vector<double> &c,
                 const std::vector<double> &d, unsigned int sz);
std::vector<double> toPowerOfTwoSize(const std::vector<double> &mtx,
                                     unsigned *power, unsigned int sz);
std::vector<double> toPowerOfTwoSize(const std::vector<double> &mtx,
                                     unsigned int sz);
std::vector<double> matrixReduce(const std::vector<double> &mtx,
                                 unsigned int sz);
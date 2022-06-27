#pragma once
#include <omp.h>
#include <math.h>

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

std::vector<double> SequentinalMultiMatrix(const std::vector<double> &A,
                                           const std::vector<double> &B, int n);
std::vector<double> CannonMultiplication(std::vector<double> A,
                                         std::vector<double> B, int BlockSize);
std::vector<double> parallelCannonMult(const std::vector<double> &A,
                                       const std::vector<double> &B, int Size);
bool Difference(const double &x, const double &y);
bool DifferenceMatrix(const std::vector<double> &A,
                      const std::vector<double> &B);
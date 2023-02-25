#pragma once

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <random>

template<typename T>
std::vector<T> regularMultiplication(const std::vector<T>& a,
    const std::vector<T>& b,
    size_t sz) {
    std::vector<T> res(sz * sz);
    for (size_t i = 0; i < sz; i++)
        for (size_t j = 0; j < sz; j++) {
            for (size_t k = 0; k < sz; k++)
                res[j + i * sz] += a[i * sz + k] * b[j + k * sz];
        }
    return res;
}

template<typename T>
std::vector<T> sum(const std::vector<T>& a,
    const std::vector<T>& b, size_t sz) {
    size_t size = sz * sz;
    std::vector<T> res(size);
    for (size_t i = 0; i < size; i++) {
        res[i] = a[i] + b[i];
    }
    return res;
}

template<typename T>
std::vector<T> diff(const std::vector<T>& a,
    const std::vector<T>& b, size_t sz) {
    size_t size = sz * sz;
    std::vector<T> res(size);
    for (size_t i = 0; i < size; i++) {
        res[i] = a[i] - b[i];
    }
    return res;
}

template<typename T>
void splitMatrix(const std::vector<T>& mtx, std::vector<T>* a,
    std::vector<T>* b, std::vector<T>* c,
    std::vector<T>* d, size_t size) {
    size_t sz = size / 2;
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = i * 2 * sz; j < i * 2 * sz + sz; j++)
            a->push_back(mtx[j]);
        for (size_t j = i * 2 * sz + sz; j < i * 2 * sz + 2 * sz; j++)
            b->push_back(mtx[j]);
    }
    for (size_t i = sz; i < 2 * sz; i++) {
        for (size_t j = i * 2 * sz; j < i * 2 * sz + sz; j++)
            c->push_back(mtx[j]);
        for (size_t j = i * 2 * sz + sz; j < i * 2 * sz + 2 * sz; j++)
            d->push_back(mtx[j]);
    }
}

template<typename T>
void mergeMatrix(std::vector<T>* mtx, const std::vector<T>& a,
    const std::vector<T>& b, const std::vector<T>& c,
    const std::vector<T>& d, size_t size) {
    size_t sz = size / 2;
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = i * sz; j < i * sz + sz; j++) mtx->push_back(a[j]);
        for (size_t j = i * sz; j < i * sz + sz; j++) mtx->push_back(b[j]);
    }
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = i * sz; j < i * sz + sz; j++) mtx->push_back(c[j]);
        for (size_t j = i * sz; j < i * sz + sz; j++) mtx->push_back(d[j]);
    }
}

template<typename T>
std::vector<T> toPowerOfTwoSize(const std::vector<T>& mtx,
    size_t sz) {
    size_t power = 2;
    while (sz > power) power *= 2;
    std::vector<T> res;
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) res.push_back(mtx[j + i * sz]);
        for (size_t j = sz; j < power; j++) res.push_back(0);
    }
    for (size_t i = sz; i < power; i++) {
        for (size_t j = 0; j < power; j++) res.push_back(0);
    }
    return res;
}

template<typename T>
std::vector<T> toPowerOfTwoSize(const std::vector<T>& mtx,
    size_t* power, size_t sz) {
    *power = 2;
    while (sz > *power) *power *= 2;
    size_t tpow = *power;
    std::vector<T> res;
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) res.push_back(mtx[j + i * sz]);
        for (size_t j = sz; j < tpow; j++) res.push_back(0);
    }
    for (size_t i = sz; i < tpow; i++) {
        for (size_t j = 0; j < tpow; j++) res.push_back(0);
    }
    return res;
}

template<typename T>
std::vector<T> matrixReduce(const std::vector<T>& mtx,
    size_t sz) {
    std::vector<T> res(sz * sz);
    size_t mtxsize = static_cast<size_t>(std::sqrt(mtx.size()));

    for (size_t i = 0; i < sz; i++)
        for (size_t j = 0; j < sz; j++) {
            res[i * sz + j] = mtx[i * mtxsize + j];
        }

    return res;
}

template<typename T>
std::vector<T> strassenMultiplication_NoCastToPowerOfTwo_omp(
    const std::vector<T>& a_, const std::vector<T>& b_,
    size_t sz) {
    std::vector<T> res;

    std::vector<T> a11, a12, a21, a22;
    std::vector<T> b11, b12, b21, b22;
    std::vector<T> p1, p2, p3, p4, p5, p6, p7;

    size_t split_sz = sz / 2;

    splitMatrix(a_, &a11, &a12, &a21, &a22, sz);
    splitMatrix(b_, &b11, &b12, &b21, &b22, sz);

    if (a11.size() > 256) {
#pragma omp parallel
        {
#pragma omp sections
            {
#pragma omp section
                p1 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    sum(a11, a22, split_sz), sum(b11, b22, split_sz), split_sz);
#pragma omp section
                p2 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    sum(a21, a22, split_sz), b11, split_sz);
#pragma omp section
                p3 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    a11, diff(b12, b22, split_sz), split_sz);
#pragma omp section
                p4 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    a22, diff(b21, b11, split_sz), split_sz);
#pragma omp section
                p5 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    sum(a11, a12, split_sz), b22, split_sz);
#pragma omp section
                p6 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    diff(a21, a11, split_sz), sum(b11, b12, split_sz), split_sz);
#pragma omp section
                p7 = strassenMultiplication_NoCastToPowerOfTwo_omp(
                    diff(a12, a22, split_sz), sum(b21, b22, split_sz), split_sz);
            }
        }
    }
    else {
        p1 = regularMultiplication(sum(a11, a22, split_sz), sum(b11, b22, split_sz),
            split_sz);
        p2 = regularMultiplication(sum(a21, a22, split_sz), b11, split_sz);
        p3 = regularMultiplication(a11, diff(b12, b22, split_sz), split_sz);
        p4 = regularMultiplication(a22, diff(b21, b11, split_sz), split_sz);
        p5 = regularMultiplication(sum(a11, a12, split_sz), b22, split_sz);
        p6 = regularMultiplication(diff(a21, a11, split_sz),
            sum(b11, b12, split_sz), split_sz);
        p7 = regularMultiplication(diff(a12, a22, split_sz),
            sum(b21, b22, split_sz), split_sz);
    }

    std::vector<T> c11, c12, c21, c22;

    c11 = sum(diff(sum(p1, p4, split_sz), p5, split_sz), p7, split_sz);
    c12 = sum(p3, p5, split_sz);
    c21 = sum(p2, p4, split_sz);
    c22 = sum(sum(diff(p1, p2, split_sz), p3, split_sz), p6, split_sz);

    mergeMatrix(&res, c11, c12, c21, c22, sz);

    return res;
}

template<typename T>
std::vector<T> strassenMultiplication_omp(const std::vector<T>& a,
    const std::vector<T>& b,
    size_t sz) {
    sz = sqrt(sz);
    size_t tpow;
    std::vector<T> a_ = toPowerOfTwoSize(a, &tpow, sz);
    std::vector<T> b_ = toPowerOfTwoSize(b, sz);
    std::vector<T> res;

    res = strassenMultiplication_NoCastToPowerOfTwo_omp(a_, b_, tpow);

    return matrixReduce(res, sz);
}

template<typename T>
std::vector<T> strassenMultiplication_NoCastToPowerOfTwo(
    const std::vector<T>& a_, const std::vector<T>& b_,
    size_t sz) {
    std::vector<T> res;

    std::vector<T> a11, a12, a21, a22;
    std::vector<T> b11, b12, b21, b22;

    splitMatrix(a_, &a11, &a12, &a21, &a22, sz);
    splitMatrix(b_, &b11, &b12, &b21, &b22, sz);

    size_t split_sz = sz / 2;

    std::vector<T> p1, p2, p3, p4, p5, p6, p7;

    if (a11.size() > 256) {
        p1 = strassenMultiplication_NoCastToPowerOfTwo(
            sum(a11, a22, split_sz), sum(b11, b22, split_sz), split_sz);
        p2 = strassenMultiplication_NoCastToPowerOfTwo(sum(a21, a22, split_sz), b11,
            split_sz);
        p3 = strassenMultiplication_NoCastToPowerOfTwo(
            a11, diff(b12, b22, split_sz), split_sz);
        p4 = strassenMultiplication_NoCastToPowerOfTwo(
            a22, diff(b21, b11, split_sz), split_sz);
        p5 = strassenMultiplication_NoCastToPowerOfTwo(sum(a11, a12, split_sz), b22,
            split_sz);
        p6 = strassenMultiplication_NoCastToPowerOfTwo(
            diff(a21, a11, split_sz), sum(b11, b12, split_sz), split_sz);
        p7 = strassenMultiplication_NoCastToPowerOfTwo(
            diff(a12, a22, split_sz), sum(b21, b22, split_sz), split_sz);
    }
    else {
        p1 = regularMultiplication(sum(a11, a22, split_sz), sum(b11, b22, split_sz),
            split_sz);
        p2 = regularMultiplication(sum(a21, a22, split_sz), b11, split_sz);
        p3 = regularMultiplication(a11, diff(b12, b22, split_sz), split_sz);
        p4 = regularMultiplication(a22, diff(b21, b11, split_sz), split_sz);
        p5 = regularMultiplication(sum(a11, a12, split_sz), b22, split_sz);
        p6 = regularMultiplication(diff(a21, a11, split_sz),
            sum(b11, b12, split_sz), split_sz);
        p7 = regularMultiplication(diff(a12, a22, split_sz),
            sum(b21, b22, split_sz), split_sz);
    }

    std::vector<T> c11, c12, c21, c22;

    c11 = sum(diff(sum(p1, p4, split_sz), p5, split_sz), p7, split_sz);
    c12 = sum(p3, p5, split_sz);
    c21 = sum(p2, p4, split_sz);
    c22 = sum(sum(diff(p1, p2, split_sz), p3, split_sz), p6, split_sz);

    mergeMatrix(&res, c11, c12, c21, c22, sz);

    return res;
}

template<typename T>
std::vector<T> strassenMultiplication(const std::vector<T>& a,
    const std::vector<T>& b,
    size_t sz) {
    sz = sqrt(sz);
    size_t tpow;
    std::vector<T> a_ = toPowerOfTwoSize(a, &tpow, sz);
    std::vector<T> b_ = toPowerOfTwoSize(b, sz);

    std::vector<T> res =
        strassenMultiplication_NoCastToPowerOfTwo(a_, b_, tpow);

    return matrixReduce(res, sz);
}
#pragma once
#include <bits/stdc++.h>
#include <stdint.h>
#include "fp8.hpp"

using namespace std;

extern void gemv(uint8_t *matrix, int row, int col, uint8_t *vector, uint8_t *result, int t);

class Linear{
private:
    
    // enum CBLAS_ORDER order = CblasRowMajor;  // 矩阵的存储顺序
    // enum CBLAS_TRANSPOSE trans = CblasNoTrans;  // 矩阵转置标志
public:
    int in_features,out_features;
    fp8 *mat;
    Linear(int in,int out,fp8 *_mat);
    ~Linear();
    void mul(fp8 *src,fp8 *dst);
};

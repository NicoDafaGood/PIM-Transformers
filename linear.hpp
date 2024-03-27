#pragma once
#include <bits/stdc++.h>
#include "fp8.hpp"
using namespace std;

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

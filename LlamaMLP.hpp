#pragma once
#include <bits/stdc++.h>
#include <cblas.h>
#include "linear.hpp"
#include "fp8.hpp"
using namespace std;

class LlamaMLP{
private:
    
public:
    Linear up_proj,down_proj,gate_proj;
    int in_features,out_features,tot;
    fp8 *up,*gate;
    LlamaMLP(int in,int out,fp8 *_up,fp8 *_gate,fp8 *_down);
    ~LlamaMLP();
    void forward(fp8 *src,fp8 *dst);
};


void SiLU(fp8 *src,int dim);
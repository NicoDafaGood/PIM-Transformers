#pragma once
#include <bits/stdc++.h>
#include "linear.hpp"
#include "fp8.hpp"
using namespace std;

class LlamaAttention{
private:
    
public:
    Linear q_proj,k_proj,v_proj,o_proj;
    int head_num,head_dim,tot;
    fp8 *K[32],*V[32],*attn[32];
    LlamaAttention(int _head_num,int _head_dim,fp8 *_Q,fp8 *_K,fp8 *_V,fp8 *_O);
    ~LlamaAttention();
    void forward(fp8 *src,fp8 *dst);
};
void Attention_kernel(fp8 *q,fp8 *K[],int n,int head_num,int head_dim,fp8 *dst[]);
void RoPE(fp8 *src,fp8 * dst,int dim,int n,fp8 theta);
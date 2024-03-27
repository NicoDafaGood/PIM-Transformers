#pragma once
#include <bits/stdc++.h>
#include"fp8.hpp"
#include "LlamaDecoderLayer.hpp"

using namespace std;

class LlamaModel{
private:
    
public:
    static const int layers_num = 32; 
    int head_dim,head_num;
    LlamaDecoderLayer *decoders[layers_num];
    LlamaModel(fp8 *Qs[],fp8 *Ks[],fp8 *Vs[],fp8 *Os[],fp8 *UPs[],fp8 *GATEs[],fp8 *DOWNs[],fp8 *attn_norms[],fp8 *ffn_norms[],int head_num,int head_dim, int mlp_dim);
    ~LlamaModel();
    void forward(fp8 *src,fp8 *dst);
};


#pragma once
#include <bits/stdc++.h>
#include "LlamaAttention.hpp"
#include "LlamaMLP.hpp"
#include "fp8.hpp"

using namespace std;

class LlamaDecoderLayer{
private:
    
public:
    LlamaAttention Attn;
    LlamaMLP MLP;
    int head_dim,head_num;
    fp8 attn_norms[4096],ffn_norms[4096];
    LlamaDecoderLayer(int head_dim,int head_num,int mlp_dim,fp8 * w_q,fp8 *w_k,fp8 *w_v,fp8 *w_o,fp8 *w_up,fp8 *w_gate,fp8 *w_down,fp8 *_attn_norms,fp8 *_ffn_norms);
    ~LlamaDecoderLayer();
    void forward(fp8 *src, fp8 *dst);
};

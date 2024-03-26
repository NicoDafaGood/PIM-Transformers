#include "LlamaMLP.hpp"

using namespace std;

void SiLU(fp8 *src,int dim)
{
    fp8 one = fp32_to_fp8(1.0);
    for(int i = 0 ;i < dim ;i++)
        src[i] = src[i] * (one/(one+exp(neg(src[i]))));
}

LlamaMLP::LlamaMLP(int in,int out,fp8 *_up,fp8 *_gate,fp8 *_down):up_proj(in,out,_up),gate_proj(in,out,_gate),down_proj(out,in,_down),in_features(in),out_features(out)
{
    // up = (fp8 *) malloc(4096*4096*sizeof(fp8));
    // gate = (fp8 *) malloc(4096*4096*sizeof(fp8));
    // puts("MLP down mat");
    // for(int i = 0;i<128;i++)
    //     cout<<_down[i]<<" ";
    // puts("");
}

LlamaMLP::~LlamaMLP()
{
    free(up);
    free(gate);
}

void LlamaMLP::forward(fp8* src,fp8 *dst)
{
    static fp8 up[11008],gate[11008],temp[11008];

    up_proj.mul(src,up);

    gate_proj.mul(src,gate);
    SiLU(gate,out_features);
    
    for(int i = 0;i<out_features;i++)
        temp[i] = up[i] * gate[i];

    down_proj.mul(temp,dst);
    

}
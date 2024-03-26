#include"LlamaDecoderLayer.hpp"

using namespace std;

LlamaDecoderLayer::LlamaDecoderLayer(int _head_dim,int _head_num,int mlp_dim,fp8 * w_q,fp8 *w_k,fp8 *w_v,fp8 *w_o,fp8 *w_up,fp8 *w_gate,fp8 *w_down,fp8 *_attn_norms,fp8 *_ffn_norms):Attn(_head_num,_head_dim,w_q,w_k,w_v,w_o), MLP(_head_num * _head_dim,mlp_dim,w_up,w_gate,w_down),head_dim(_head_dim),head_num(_head_num)
{
    memcpy(attn_norms,_attn_norms,4096*sizeof(fp8));
    memcpy(ffn_norms,_ffn_norms,4096*sizeof(fp8));

}

void RMS(fp8 *mat,fp8 *src,int dim,fp8 *dst)
{

    fp8 eps = fp32_to_fp8(1e-5);
    fp8 sum = fp32_to_fp8(0);
    for(int i = 0;i<dim;i++)
    {
        sum = sum + src[i] * src[i];
        // cout<<src[i] * src[i]<<" ";
    }
    sum = sum/fp32_to_fp8((float)dim);
    sum = sqr(sum+eps);
    
    for(int i = 0;i<dim;i++)
        dst[i] = src[i] / sum * mat[i];

}

LlamaDecoderLayer::~LlamaDecoderLayer(){

}

void LlamaDecoderLayer::forward(fp8 * src,fp8 *dst)
{
    // cout<<"src: "<<src[0]<<endl;
    // for(int i = 0;i < head_dim;i++)
    //     cout<<src[i]<<" ";
    // puts("");
    static fp8 in[4096],mid[4096],out[4096],temp[4096];
    for(int i = 0;i<head_dim * head_num;i++)
        in[i] = src[i];
    RMS(attn_norms,in,head_dim * head_num,temp);
    // cout<<"RMS1: "<<temp[0]<<endl;

    Attn.forward(temp,mid);

    for(int i=0;i<head_dim * head_num;i++)
        mid[i] = mid[i] + in[i];
    RMS(ffn_norms,mid,head_dim * head_num,temp);
    // cout<<"RMS2: "<<temp[0]<<endl;

    
    MLP.forward(temp,out);


    for(int i = 0;i < head_dim * head_num;i++)
        out[i] = out[i] + mid[i];
    
    for(int i = 0;i < head_dim * head_num;i++)
    {
        dst[i] = out[i];
    }

    // cout<<dst[0]<<endl;
    // for(int i = 0;i < head_dim;i++)
    //     cout<<dst[i]<<" ";
}
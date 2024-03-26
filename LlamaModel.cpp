#include "LlamaModel.hpp"

using namespace std;

LlamaModel::LlamaModel(fp8 *Qs[],fp8 *Ks[],fp8 *Vs[],fp8 *Os[],fp8 *UPs[],fp8 *GATEs[],fp8 *DOWNs[],fp8 *attn_norms[],fp8 *ffn_norms[],int _head_num,int _head_dim, int mlp_dim) : head_dim(_head_dim),head_num(_head_num)
{
    for(int i = 0;i<layers_num;i++)
        decoders[i] = new LlamaDecoderLayer(_head_dim,_head_num,mlp_dim,Qs[i],Ks[i],Vs[i],Os[i],UPs[i],GATEs[i],DOWNs[i],attn_norms[i],ffn_norms[i]);
}
LlamaModel::~LlamaModel()
{
    for(int i = 0;i<layers_num;i++)
        delete decoders[i];
}
void LlamaModel::forward(fp8 *src,fp8 *dst)
{
    fp8 temp[2][4096];
    int in = 1,out = 0;
    memcpy(temp[in],src,head_dim*head_num*sizeof(fp8));
    for(int i = 0;i<layers_num;i++)
    {
        decoders[i]->forward(temp[in],temp[out]);
        swap(in,out);
    }
    memcpy(dst,temp[in],head_dim*head_num*sizeof(fp8));
}
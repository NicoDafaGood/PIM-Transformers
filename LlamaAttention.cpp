#include "LlamaAttention.hpp"
using namespace std;



void RoPE(fp8 *src,fp8 * dst,int dim,int n,double theta = 10000.0)
{
    static fp8 half_rot[128],c[128],s[128];
    for(int i =0;i < dim/2;i++)
    {
        float angle = pow(theta,-i*2.0/dim) * n;
        s[i] = fp32_to_fp8((float)sin(angle));
        c[i] = fp32_to_fp8((float)cos(angle));
        s[i+dim/2] = s[i];
        c[i+dim/2] = c[i];
    }
    for(int i =0;i < dim/2;i++)
    {
        half_rot[i] = neg(src[i+dim/2]);
        half_rot[i+dim/2] =  src[i];
    }

    for(int i = 0; i < dim;i++)
        dst[i]  = src[i]*c[i] + half_rot[i] * s[i];
}

void softmax(fp8 *src,int dim)
{
    static fp8 temp[4096];
    fp8 sum = fp32_to_fp8(0),mx=fp32_to_fp8(-1e9);
    for(int i = 0; i < dim; i++)
        mx = max(src[i],mx);
    for(int i = 0; i < dim; i++)
    {
        temp[i] = exp(src[i] - mx);
        sum = sum+temp[i];
    }
    for(int i = 0; i < dim; i++)
        src[i] = temp[i]/sum;
    // cout<<sum<<endl;
}

void dropout(fp8 *src,int lenth,float rate)
{
    std::random_device rd;  
    std::mt19937 gen(rd());  
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for(int i = 0;i < lenth; i++)
        if(dis(gen)<rate)
            src[i] = fp32_to_fp8(0);
}
void Attention_kernel(fp8 *q,fp8 *K[],int n,int head_num,int head_dim,fp8 *dst[])
{
    // cerr<<"ATTN\n";
    // puts("attn");
    // puts("attn");
    fp8 sqr = fp32_to_fp8((float)sqrt(head_dim));
    for(int i = 0;i<head_num;i++)
    {
        
        // cblas_sgemv(CblasRowMajor, CblasNoTrans, n, head_dim, 1.0, K[i], head_dim, q + i*head_dim, 1, 0.0, dst[i], 1);
        for(int j = 0 ;j<n;j++)
            dst[i][j] = dst[i][j]/sqr;
        
        // for(int j = 0 ;j<n;j++)
        //     cout<<dst[i][j]<<" ";
        softmax(dst[i],n);
        dropout(dst[i],n,0);
        // for(int j = 0 ;j<n;j++)
        //     cout<<dst[i][j]<<" ";
        // puts("");
    }
    // puts("");
}

LlamaAttention::LlamaAttention(int _head_num,int _head_dim,fp8 *_Q,fp8 *_K,fp8 *_V,fp8 *_O) : head_num(_head_num),head_dim(_head_dim),q_proj(_head_num * _head_dim,_head_num * _head_dim,_Q),k_proj(_head_num * _head_dim,_head_num * _head_dim,_K),v_proj(_head_num * _head_dim,_head_num * _head_dim,_V),o_proj(_head_num * _head_dim,_head_num * _head_dim,_O){
    for(int i = 0;i < head_num;i++)
    {
        K[i] = (fp8 *)malloc(128*4096*sizeof(fp8));
        V[i] = (fp8 *)malloc(128*4096*sizeof(fp8));
        attn[i] = (fp8 *)malloc(4096*sizeof(fp8));
    }
    tot = 0;
}
LlamaAttention::~LlamaAttention(){
    for(int i = 0;i < head_num;i++)
    {
        free(K[i]);
        free(V[i]);
        free(attn[i]);
    }
}

void LlamaAttention::forward(fp8 *src,fp8 *dst)
{
    static fp8 temp[4096],q[4096],k[4096];


    q_proj.mul(src,temp);
    // puts("before");
    // for(int i= 0;i<head_dim;i++)
    //     cout<<temp[i]<<" ";
    // puts("");
    for(int i =0 ;i<head_num * head_dim;i+=head_dim)
        RoPE(temp+i,q+i,head_dim,tot/(head_num * head_dim));
    // puts("after");
    // for(int i= 0;i<head_dim;i++)
    //     cout<<q[i]<<" ";
    // puts("");
    k_proj.mul(src,temp);

    for(int i =0 ;i<head_num * head_dim;i+=head_dim)
        RoPE(temp+i,k+i,head_dim,tot/(head_num * head_dim));
    
    for(int i = 0;i < head_num;i++)
        memcpy(K[i]+tot/head_num,k+i*head_dim,head_dim*sizeof(fp8));
    v_proj.mul(src,temp);
    for(int i = 0;i < head_num;i++)
        memcpy(V[i]+tot/head_num,temp+i*head_dim,head_dim*sizeof(fp8));
    // puts("value");
    // for(int i =0;i<=tot/head_num;i+=head_dim)
    //     for(int j = 0;j<head_dim;j++)
    //         cout<<V[0][i+j]<<" ";
    // puts("");
    tot += (head_num * head_dim);
    int n = tot/(head_num * head_dim);
    Attention_kernel(q,K,n,head_num, head_dim,attn);

    for(int i = 0;i < head_num;i++)
    {

        // cblas_sgemv(CblasColMajor, CblasNoTrans, head_dim, n, 1.0, V[i], head_dim, attn[i], 1, 0.0, temp + i * head_dim, 1);

    }
    o_proj.mul(temp,dst);
}
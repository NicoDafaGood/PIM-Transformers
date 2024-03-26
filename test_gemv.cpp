#include<cblas.h>
#include<bits/stdc++.h>

using namespace std;

fp8 mat[4096*11008];
fp8 src[11008],dst[11008];


int main()
{
    // int in_features = 11008;
    // int out_features = 4096;
    int in_features = 4096;
    int out_features = 11008;
    cblas_sgemv(CblasRowMajor,CblasNoTrans,out_features,in_features,1,mat,in_features,src,1,0,dst,1);
}
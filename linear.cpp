#include"linear.hpp"
#include"fp8.hpp"
using namespace std;

Linear::Linear(int in,int out,fp8 *_mat){
        in_features = in;
        out_features = out;
        mat = (fp8 *)malloc(sizeof(fp8)*in*out);
        memcpy(mat,_mat,sizeof(fp8)*in*out);
    }

Linear::~Linear(){
        free(mat);
    }

void Linear:: mul(fp8 *src,fp8 *dst){
    // cerr<<"MUL\n";
    // cout << in_features<<" "<<out_features<<endl;
    // cblas_sgemv(CblasRowMajor,CblasNoTrans,out_features,in_features,1,mat,in_features,src,1,0,dst,1);
}


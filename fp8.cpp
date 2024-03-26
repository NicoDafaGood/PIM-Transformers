#include "fp8.hpp"
#include <stdint.h>

// fp8 fp8::mul_mat[256][256];
// fp8 fp8::add_mat[256][256];
// fp8 fp8::div_mat[256][256];
// fp8 fp8::exp_mat[256];
// fp8 fp8::sqr_mat[256];
float fp8::base[256];

void fp8::init()
{
    cout<<"init\n";
    for(uint8_t i = 0; i<256;i++)
    {
        // cout<<(int)i<<" Srart\n";
        uint32_t sign =  i>>7;
        uint32_t expond = (i>>2)&(0x1f);
        uint32_t frac = (i&(0x3));
        uint32_t x = sign<<31 | ((expond + 127 - 15)<<23) | (frac<<21);
        if(expond == 0 && frac ==0)
        {
            base[i] = 0;
            continue;
        }
        if(expond == 0)
        {
            expond+=127 - 15;
            // cout<<"temp expond "<<expond<<endl;
            while(!(frac & (1<<2)))
            {
                frac<<=1;
                expond -= 1;
            }
            frac &= (0x3);
            x = sign<<31 | (expond<<23) | (frac<<21);
        }
        base[i] = *((float *)(&x));
        cout<<(int)i<<" "<<base[i]<<" End\n";
        cout<<"sign "<<(int)sign<<endl;
        cout<<"expond "<<(int)expond<<endl;
        cout<<"frac "<<(int)frac<<endl;
        if(i==255)
            break;
    }
    return;
}

fp8 operator +(fp8 A,fp8 B)
{
    return fp32_to_fp8(fp8_to_fp32(A) + fp8_to_fp32(B));
    // return fp8::add_mat[A.val][B.val];
}

fp8 operator -(fp8 A,fp8 B)
{
    return fp32_to_fp8(fp8_to_fp32(A) - fp8_to_fp32(B));;
}

fp8 operator *(fp8 A,fp8 B)
{
    return fp32_to_fp8(fp8_to_fp32(A) * fp8_to_fp32(B));;
}

fp8 operator /(fp8 A,fp8 B)
{
    return fp32_to_fp8(fp8_to_fp32(A) / fp8_to_fp32(B));;
}

bool operator <(fp8 A,fp8 B)
{
    return fp8_to_fp32(A) < fp8_to_fp32(B);

}

fp8 exp(fp8 A)
{
    return fp32_to_fp8((float)exp(fp8_to_fp32(A)));
}

fp8 sqr(fp8 A)
{
    return fp32_to_fp8((float)sqrt(fp8_to_fp32(A)));
}

fp8 neg(fp8 A)
{
    return fp32_to_fp8(-fp8_to_fp32(A));
}

float fp8_to_fp32(fp8 A)
{
    return fp8::base[A.val];
}

fp8 fp32_to_fp8(float A)
{
    int p = 1;
    if(A<0)
    {
        A = -A;
        p = -1;
    }
    unsigned char l = 0,r=127,mid;
    while(l<r)
    {
        mid = (l+r)/2;
        if(fp8::base[mid] < A)
            l = mid + 1;
        else
            r = mid - 1;
    }
    fp8 ans;
    if(p==-1)
        mid += 128;
    ans.val = mid;
    return ans;
}
#pragma once
#include <bits/stdc++.h>
using namespace std;

class fp8{
public:
    unsigned char val;
    void init(const string path);
    friend std::ostream& operator<<(std::ostream& os, const fp8& obj) {
        os << static_cast<int>(obj.val); // 将unsigned char转换为int输出
        return os;
    }
    static fp8 mul_mat[256][256];
    static fp8 add_mat[256][256];
    static fp8 div_mat[256][256];
    static fp8 sqr_mat[256];
    static fp8 exp_mat[256];
    static float base[256];
    static void init();
};



fp8 operator +(fp8 A,fp8 B);
fp8 operator -(fp8 A,fp8 B);
fp8 operator *(fp8 A,fp8 B);
fp8 operator /(fp8 A,fp8 B);
bool operator <(fp8 A,fp8 B);

// fp8 max(fp8 A,fp8 B);
// fp8 min(fp8 A,fp8 B);
fp8 exp(fp8 A);
fp8 neg(fp8 A);
fp8 sqr(fp8 A);
float fp8_to_fp32(fp8 A);
fp8 fp32_to_fp8(float A);

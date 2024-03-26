#include <bits/stdc++.h>
#include <cblas.h>
#include "LlamaModel.hpp"

using namespace std;
fp8 *Qs[32],*Ks[32],*Vs[32],*Os[32],*UPs[32],*GATEs[32],*DOWNs[32],*attn_norms[32],*ffn_norms[32];

fp8 in[4096],out[4096],temp[11008];

int main() {
    cout<<"Start"<<endl;
    fp8::init();
    cout<<"OK"<<endl;
    std::string file_path = "../LLM/model_params.bin";
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << file_path << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read file: " << file_path << std::endl;
        return 1;
    }

    float* model_params_float = reinterpret_cast<float*>(buffer.data());
    fp8* model_params_array = reinterpret_cast<fp8*>(buffer.data());
    std::size_t num_elements = size / sizeof(float);
    for(size_t i = 0;i < num_elements;i++)
    {
        model_params_array[i] = fp32_to_fp8(model_params_float[i]);
        if(i%(4096*4096) == 0)
        {
            cout<<i<<"/"<<num_elements<<"\n";
            cout<<"Origianl: "<<model_params_float[i]<<"\n";
            cout<<"New: "<<model_params_array[i]<<"\n";
            cout<<"New(to fp32): "<<fp8_to_fp32(model_params_array[i])<<"\n";
        }
    }
    std::cout<<size<<std::endl;
    // 打印数组中的元素
    // std::cout << "Array elements: ";
    // for (std::size_t i = 0; i < num_elements; ++i) {
    //     std::cout << model_params_array[i] << " ";
    // }
    std::cout << std::endl;
    size_t pt = 0;
    for(int i = 0;i < 32;i++)
    {
        Qs[i] = model_params_array + pt;
        pt += 4096*4096;
        Ks[i] = model_params_array + pt;
        pt += 4096*4096;
        Vs[i] = model_params_array + pt;
        pt += 4096*4096;
        Os[i] = model_params_array + pt;
        pt += 4096*4096;
        GATEs[i] = model_params_array + pt;
        pt += 4096*11008;
        UPs[i] = model_params_array + pt;
        pt += 4096*11008;
        DOWNs[i] = model_params_array + pt;
        pt += 4096*11008;
        attn_norms[i] = model_params_array + pt;
        pt +=4096;
        ffn_norms[i] = model_params_array + pt;
        pt +=4096;
    }
    cout << pt * sizeof(fp8)<<"\n";
    LlamaModel my_model(&Qs[0],&Ks[0],&Vs[0], &Os[0],&UPs[0],&GATEs[0],&DOWNs[0],&attn_norms[0],&ffn_norms[0],32,128,11008);
    for(int i = 0;i<4096;i++)
        in[i] = fp32_to_fp8((float)((i*i%11451)/11451.0));
    // for(int i = 0;i<4096;i++)
    //     in[i] = i;
    cout<<"mat fp8:"<<my_model.decoders[0]->Attn.q_proj.mat[0]<<endl;
    my_model.forward(in,temp);
    puts("ans");
    for(int i = 0;i<1024;i++)
        cout<<temp[i]<<" ";
    puts("");
    my_model.forward(in,temp);
    // my_model.decoders[0]->forward(in,temp);
    puts("ans");
    for(int i = 0;i<1024;i++)
        cout<<temp[i]<<" ";
    puts("");
    // puts("UP");
    // for(int i = 0;i<1024;i++)
    //     cout<<my_model.decoders[0]->MLP.up_proj.mat[i]<<" ";
    // puts("");
    // puts("GATE");
    // for(int i = 0;i<1024;i++)
    //     cout<<my_model.decoders[0]->MLP.gate_proj.mat[i]<<" ";
    // puts("");
    // puts("DOWN");
    // for(int i = 0;i<1024;i++)
    //     cout<<my_model.decoders[0]->MLP.down_proj.mat[i]<<" ";
    // puts("");
    return 0;
}
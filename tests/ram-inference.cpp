#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "direct-weights.h"
#include "routing-table.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using json=nlohmann::json;
static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static std::string model_path;
static bool gpu=false;
static size_t original_bytes=0;
static std::vector<float> run(bool direct,bool mapped,int prune,bool media=false,bool preload=false) {
    setenv("DSV4F_RAM_BUDGET_MIB",direct?"131072":"0",1);
    setenv("DSV4F_RAM_RESERVE_MIB","4",1);setenv("DSV4F_RAM_CACHE_MIB","4",1);
    setenv("DSV4F_RAM_PRELOAD",preload?"1":"0",1);
    setenv("DSV4F_ADAPTIVE_PRUNING",std::to_string(prune).c_str(),1);
    dsv4f_routing_init(model_path);
    auto mp=llama_model_default_params();mp.load_mode=mapped?LLAMA_LOAD_MODE_MMAP:LLAMA_LOAD_MODE_NONE;
    mp.n_gpu_layers=gpu?999:0;mp.use_extra_bufts=false;mp.no_host=true;
    const llama_model_tensor_buft_override overrides[]={{"\\.ffn_(gate|up|down)_exps",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    mp.tensor_buft_overrides=overrides;
    auto * model=llama_model_load_from_file(model_path.c_str(),mp);require(model,"model load");
    size_t bytes=0;
    for(int il=0;il<int(model->hparams.n_layer());++il) {
        const auto & l=model->layers[il];
        for(auto * t:{l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps})bytes+=ggml_nbytes(t);
        if(prune) {
            std::vector<float> bias(model->hparams.n_expert);
            ggml_backend_tensor_get(l.ffn_exp_probs_b_vl,bias.data(),0,bias.size()*sizeof(float));
            const int active=std::count_if(bias.begin(),bias.end(),[](float x){return std::isfinite(x);});
            require(active>=int(model->hparams.n_expert_used(il)),"vision mask retains top-k");
            if(!direct && !mapped)require(l.ffn_gate_exps->ne[2]==active && l.ffn_expert_ids,"packed expert dimensions");
            if(l.ffn_gate_tid2eid) {
                std::vector<int32_t> ids(ggml_nelements(l.ffn_gate_tid2eid));
                ggml_backend_tensor_get(l.ffn_gate_tid2eid,ids.data(),0,ids.size()*sizeof(int32_t));
                for(auto id:ids)require(id>=0 && id<int(bias.size()) && std::isfinite(bias[id]),"hash routes avoid pruned experts");
            }
        }
    }
    if(!prune)original_bytes=bytes;
    else if(!direct && !mapped)require(bytes<=original_bytes*(100-prune)/100,"resident pruning meets byte target");
    auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=64;cp.n_ubatch=32;cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=4;cp.offload_kqv=gpu;cp.op_offload=gpu;
    cp.type_k=cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx=llama_init_from_model(model,cp);require(ctx,"context");
    if(preload){const auto result=dsv4f_preload_experts();require(result.planes>0 && result.budget_limited,"bounded startup preload");}
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));std::vector<float> logits;
    const int stop=media?3:36;
    for(int pos=0;pos<stop;) {
        const int n=media?stop:pos==0?32:pos==32?3:1;
        const int embd=media?model->hparams.n_embd:0;
        auto batch=llama_batch_init(n,embd,1);batch.n_tokens=n;
        for(int i=0;i<n;++i) {
            if(media)for(int e=0;e<embd;++e)batch.embd[i*embd+e]=.03f*std::sin(float(e+i));
            else batch.token[i]=2+(pos+i)%30;
            batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;
        }
        require(llama_decode(ctx,batch)==0,"prefill/decode");
        for(int i=0;i<n;++i){const auto * row=llama_get_logits_ith(ctx,i);logits.insert(logits.end(),row,row+nv);}
        llama_batch_free(batch);pos+=n;
    }
    llama_free(ctx);llama_model_free(model);return logits;
}
static void compare(const std::vector<float> & a,const std::vector<float> & b) {
    require(!a.empty() && a.size()==b.size(),"logit shapes");double error=0,norm=0;
    for(size_t i=0;i<a.size();++i){require(std::isfinite(a[i]) && std::isfinite(b[i]),"finite logits");error+=std::pow(double(a[i])-b[i],2);norm+=double(a[i])*a[i];}
    const double relative=std::sqrt(error/std::max(norm,1e-30));std::fprintf(stderr,"logit relative RMSE=%g\n",relative);
    // Generic batched CPU matmul and complete-expert Q8 activation paths
    // round differently; the direct cache bytes are verified exactly separately.
    require(relative<2e-3,"resident/direct/packed inference parity");
}
int main(int argc,char ** argv) {
    require(argc>=2,"fixture argument");model_path=std::filesystem::absolute(argv[1]).string();gpu=argc>2;
    setenv("DSV4F_MTP","0",1);setenv("DSV4F_ADAPTIVE_ROUTING","0",1);
    setenv("DSV4F_WHOLE_EXPERT","1",1);setenv("DSV4F_GPU_SERVICE","1",1);
    setenv("DSV4F_GPU_WAIT","callback",1);setenv("GGML_CUDA_MOE_CACHE","0",1);
    setenv("DSV4F_PREFILL_STREAM","1",1);setenv("DSV4F_PREFILL_GROUP","2",1);
    setenv("DSV4F_EXPERT_CACHE_MIB","4",1);setenv("DSV4F_ROUTER_FUSED","1",1);
    ggml_backend_load_all();if(gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))return 77;
    if(!gpu)ggml_dsv4f_experts={};
    const auto previous=std::filesystem::current_path();char temp[]="/tmp/dsv4f-ram-inference-XXXXXX";require(mkdtemp(temp),"temporary directory");
    std::filesystem::current_path(temp);std::filesystem::create_directory("models");
    json history={{"version",1},{"model",std::filesystem::path(model_path).stem().string()},{"experts",json::object()}};
    for(int l=0;l<3;++l)for(int id=0;id<8;++id)history["experts"]["blk."+std::to_string(l)+".ffn_exps.weight"][std::to_string(id)]=(id*7+l*3)%8+1;
    {std::ofstream f(std::filesystem::path("models")/(std::filesystem::path(model_path).stem().string()+".routing.json"));f<<history;}
    // The server initializes history for the target once, then loads the
    // MTP-only file. Its absent trunk must not enter the pruning planner.
    for(int prune:{0,25})for(const char * budget:{"0","131072"}) {
        setenv("DSV4F_RAM_BUDGET_MIB",budget,1);setenv("DSV4F_ADAPTIVE_PRUNING",std::to_string(prune).c_str(),1);
        dsv4f_routing_init(model_path);
        auto mp=llama_model_default_params();mp.no_alloc=true;mp.load_mtp=true;mp.n_gpu_layers=0;
        mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.use_extra_bufts=false;
        auto mtp_path=std::filesystem::path(model_path).parent_path()/"V4-MTP.gguf";
        auto * mtp=llama_model_load_from_file(mtp_path.c_str(),mp);require(mtp,"MTP-only model loading");
        require(!mtp->layers[0].ffn_gate_exps && mtp->layers[3].ffn_gate_exps &&
                mtp->layers[3].ffn_gate_exps->ne[2]==8 && !mtp->layers[3].ffn_expert_ids,"MTP experts excluded from pruning");
        llama_model_free(mtp);
    }
    const auto baseline=run(false,false,0);compare(baseline,run(true,false,0));compare(baseline,run(true,false,0,false,true));
    for(int pct:{25,50}) {
        const auto mapped=run(false,true,pct);compare(mapped,run(false,false,pct));compare(mapped,run(true,false,pct));
        const auto media=run(false,true,pct,true);compare(media,run(false,false,pct,true));
    }
    std::filesystem::current_path(previous);std::filesystem::remove_all(temp);
    std::puts("RAM tiering, eviction, preload, resident/mmap pruning, hash/vision routing and logits PASS");
}

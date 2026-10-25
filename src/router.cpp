#include "moe.h"
#include "ggml-backend-moe-cache.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// Keep the generic graph for prefill and other backends. This packed operation
// exposes both routing outputs while retaining an independent CPU fallback.
static void router_cpu(ggml_tensor * dst,int ith,int,void *) {
    if(ith)return;
    const auto * logits=dst->src[0],*bias=dst->src[1];
    const int ne=logits->ne[0],nk=dst->ne[0],nt=dst->ne[1];
    float scale;std::memcpy(&scale,&dst->op_params[12],4);
    std::vector<float> scores(ne),rank(ne);std::vector<int32_t> order(ne);
    for(int t=0;t<nt;++t) {
        for(int i=0;i<ne;++i) {
            const float x=*(const float *)((const char *)logits->data+t*logits->nb[1]+i*4);
            scores[i]=std::sqrt(x>20.f?x:std::log(1.f+std::exp(x)));
            rank[i]=scores[i]+((const float *)bias->data)[i];
            if(std::isnan(rank[i]))rank[i]=-std::numeric_limits<float>::max();
            order[i]=i;
        }
        std::partial_sort(order.begin(),order.begin()+nk,order.end(),[&](int a,int b){return rank[a]>rank[b] || (rank[a]==rank[b] && a<b);});
        double sum=0;for(int k=0;k<nk;++k)sum+=scores[order[k]];
        const float denominator=std::max(float(sum),6.103515625e-5f);
        for(int k=0;k<nk;++k) {
            const size_t index=size_t(t)*nk+k;
            std::memcpy((char *)dst->data+index*4,&order[k],4);
            const float weight=scale*(dst->op_params[10]?scores[order[k]]/denominator:scores[order[k]])+0.f;
            std::memcpy((char *)dst->data+(size_t(nk)*nt+index)*4,&weight,4);
        }
    }
}
void dsv4f_router(ggml_context * ctx,ggml_tensor * logits,ggml_tensor * bias,
                 int nk,bool normalize,float scale,ggml_tensor ** ids,ggml_tensor ** probabilities) {
    GGML_ASSERT(logits->type==GGML_TYPE_F32 && logits->nb[0]==4 &&
                logits->ne[0]>0 && logits->ne[0]<=512 && logits->ne[1]>0 && logits->ne[1]<=4 &&
                logits->ne[2]==1 && logits->ne[3]==1);
    GGML_ASSERT(bias->type==GGML_TYPE_F32 && ggml_is_contiguous(bias) && ggml_nelements(bias)==logits->ne[0]);
    GGML_ASSERT(nk>0 && nk<=32 && nk<=logits->ne[0]);
    ggml_tensor * inputs[]={logits,bias};const int nt=logits->ne[1];
    auto * packed=ggml_custom_4d(ctx,GGML_TYPE_F32,nk,nt,2,1,inputs,2,router_cpu,1,nullptr);
    packed->op_params[9]=GGML_DSV4F_ROUTER_TAG;packed->op_params[10]=normalize;
    std::memcpy(&packed->op_params[12],&scale,4);ggml_set_name(packed,"dsv4f_router");
    *ids=ggml_view_2d(ctx,packed,nk,nt,nk*4,0);(*ids)->type=GGML_TYPE_I32;
    *probabilities=ggml_view_3d(ctx,packed,1,nk,nt,4,nk*4,size_t(nk)*nt*4);
}

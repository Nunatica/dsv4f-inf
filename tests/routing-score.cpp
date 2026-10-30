#include "ggml-dsv4f-routing.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static uint64_t reference(intptr_t count) {
    if(count<=0)return 0;
    uint64_t value=uint64_t(count)+1,bonus=0;
    for(;value>1;value>>=1)++bonus;
    return bonus;
}
static void check(intptr_t count) {
    const auto expected=reference(count),actual=ggml_dsv4f_routing_bonus(count);
    if(actual!=expected) {
        std::fprintf(stderr,"routing bonus mismatch: count=%lld actual=%llu expected=%llu\n",
            (long long)count,(unsigned long long)actual,(unsigned long long)expected);
        std::exit(1);
    }
}
int main() {
    check(INTPTR_MIN);check(-1);check(0);check(INTPTR_MAX);
    // Every power-of-two transition of log2(count + 1), including the signed
    // pointer maximum, must retain exactly the same admission/eviction score.
    for(unsigned bit=0;bit<sizeof(intptr_t)*8-1;++bit) {
        const intptr_t power=intptr_t(uintptr_t(1)<<bit);
        check(power);check(power-1);if(power>1)check(power-2);
    }
    for(intptr_t n=0;n<65536;++n)check(n);
    uint64_t state=0x4d595df4d0f33173ULL;
    for(int i=0;i<1000000;++i) {
        state^=state<<13;state^=state>>7;state^=state<<17;
        check(intptr_t(state&uint64_t(INTPTR_MAX)));
    }
    std::puts("Routing score parity: signed limits, every power-of-two transition, small counts and 1M random values PASS");
}

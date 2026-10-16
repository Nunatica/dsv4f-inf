#include "ggml-dsv4f-budget.h"
#include <cstdio>
#include <limits>

int main() {
    constexpr size_t mib = 1ULL << 20;
    bool ok = true;
    auto check = [&](const char * name, size_t actual, size_t expected) {
        if (actual != expected) { std::fprintf(stderr, "%s: got %zu, expected %zu\n", name, actual, expected); ok = false; }
    };
    check("temporary allocation fits", ggml_dsv4f_vram_deficit(1024*mib,512*mib,256*mib),0);
    check("exact headroom", ggml_dsv4f_vram_deficit(1024*mib,512*mib,512*mib),0);
    check("temporary allocation needs initial cache", ggml_dsv4f_vram_deficit(600*mib,512*mib,256*mib),168*mib);
    check("phase already below headroom", ggml_dsv4f_vram_deficit(425*mib,512*mib,0),87*mib);
    check("temporary allocation preserves library allowance", ggml_dsv4f_vram_deficit(900*mib,512*mib+ggml_dsv4f_cache_guard,256*mib),124*mib);
    check("MTP workspace replenishes guard", ggml_dsv4f_vram_deficit(700*mib,512*mib+ggml_dsv4f_cache_guard,0),68*mib);
    check("allocation sum overflow", ggml_dsv4f_vram_deficit(0,512*mib,std::numeric_limits<size_t>::max()),std::numeric_limits<size_t>::max());
    // 11 GiB left after target+MTP, 1 GiB requested headroom, 256 MiB guard,
    // 16 MiB complete experts: 10 GiB requested must shrink to 624 experts.
    check("MTP accounted before cache", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 10240*mib, 16*mib, 2000), 624);
    check("other application uses 2 GiB", ggml_dsv4f_cache_slots(9216*mib, 1024*mib, 10240*mib, 16*mib, 2000), 496);
    check("custom 512 MiB guard", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 10240*mib, 16*mib, 2000, 0, 512*mib), 608);
    check("disabled guard", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 10240*mib, 16*mib, 2000, 0, 0), 640);
    check("explicit smaller ceiling", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 2048*mib, 16*mib, 2000), 128);
    check("model expert limit", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 10240*mib, 16*mib, 100), 100);
    check("disabled", ggml_dsv4f_cache_slots(11264*mib, 1024*mib, 0, 16*mib, 2000), 0);
    check("extension shares ceiling", ggml_dsv4f_cache_remaining(10240*mib,8192*mib),2048*mib);
    check("base at ceiling", ggml_dsv4f_cache_remaining(10240*mib,10240*mib),0);
    check("base over reduced ceiling", ggml_dsv4f_cache_remaining(8192*mib,10240*mib),0);
    check("automatic extension", ggml_dsv4f_cache_remaining(std::numeric_limits<size_t>::max(),8192*mib),std::numeric_limits<size_t>::max());
    check("insufficient headroom", ggml_dsv4f_cache_slots(512*mib, 1024*mib, 10240*mib, 16*mib, 2000), 0);
    check("guard consumes remainder", ggml_dsv4f_cache_slots(1280*mib, 1024*mib, 10240*mib, 16*mib, 2000), 0);
    check("slot metadata", ggml_dsv4f_cache_slots(1296*mib, 1024*mib, 10240*mib, 16*mib, 2000, 4), 0);
    check("overflow", ggml_dsv4f_cache_slots(std::numeric_limits<size_t>::max(), 0, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), 2000, 4), 0);
    std::puts(ok ? "VRAM cache budget PASS" : "VRAM cache budget FAIL");
    return ok ? 0 : 1;
}

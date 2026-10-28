#include "llama-model-loader.h"
#include "ggml-backend-impl.h"
#include "gguf.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unistd.h>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct fixture {
    std::string path;
    fixture(bool invalid = false, bool large = false) {
        char name[] = "/tmp/dsv4f-load-pipeline-XXXXXX";
        const int fd = mkstemp(name);
        require(fd >= 0, "mkstemp failed");
        close(fd);
        path = name;
        ggml_context_ptr ctx(ggml_init({1 << 20, nullptr, false}));
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", "llama");
        for (int i = 0; i < 4; ++i) {
            auto * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, i == 2 ? (large ? 8192 : 80) : 16);
            ggml_set_name(t, ("weight_" + std::to_string(i)).c_str());
            for (int j = 0; j < t->ne[0]; ++j) {
                static_cast<float *>(t->data)[j] = float(100 * i + j);
            }
            if (invalid && i == 2) {
                static_cast<float *>(t->data)[0] = std::numeric_limits<float>::quiet_NaN();
            }
            gguf_add_tensor(meta.get(), t);
        }
        require(gguf_write_to_file(meta.get(), path.c_str(), false), "write GGUF failed");
    }
    ~fixture() { std::remove(path.c_str()); }
};

struct probe {
    std::mutex mutex;
    std::condition_variable changed;
    size_t allocations = 0;
    bool allocating_second = false;
    bool overlap = false;
    bool check_overlap = false;
    bool fail_allocation = false;
    float cancel_at = 2.0f;
    std::vector<float> progress;
};

static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t buft, size_t size) {
    auto & p = *static_cast<probe *>(buft->context);
    std::unique_lock<std::mutex> lock(p.mutex);
    ++p.allocations;
    if (p.allocations == 2 && p.check_overlap) {
        p.allocating_second = true;
        p.changed.notify_all();
        require(p.changed.wait_for(lock, std::chrono::seconds(5), [&] { return p.overlap; }),
                "reader did not overlap the second allocation");
    }
    if (p.allocations == 2 && p.fail_allocation) {
        return nullptr;
    }
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}

static bool progress(float value, void * data) {
    auto & p = *static_cast<probe *>(data);
    std::unique_lock<std::mutex> lock(p.mutex);
    if (p.progress.empty() && p.check_overlap) {
        require(p.changed.wait_for(lock, std::chrono::seconds(5), [&] { return p.allocating_second; }),
                "second allocation waited for the first read to finish");
        p.overlap = true;
        p.changed.notify_all();
    }
    require(p.progress.empty() || value >= p.progress.back(), "progress went backwards");
    p.progress.push_back(value);
    return value < p.cancel_at;
}

static void run(const std::string & mode) {
    // Make the truncated-file case larger than stdio's cached header block.
    fixture file(mode == "invalid", mode == "read-error");
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, file.path, splits, nullptr,
            LLAMA_LOAD_MODE_NONE, true, false, false, nullptr, nullptr);
    loader.init_mappings();
    if (mode == "read-error") {
        require(truncate(file.path.c_str(), 0) == 0, "truncate failed");
    }
    auto * ctx = loader.contexts.front().get();
    probe p;
    p.check_overlap = mode == "success" || mode == "allocation-error";
    p.fail_allocation = mode == "allocation-error";
    if (mode == "cancel") { p.cancel_at = 0.2f; }
    if (mode == "cancel-final") { p.cancel_at = 1.0f; }
    if (mode == "cancel-first") { p.cancel_at = 0.0f; }

    auto buft = *ggml_backend_cpu_buffer_type();
    buft.context = &p;
    buft.iface.alloc_buffer = allocate;
    std::vector<ggml_backend_buffer_ptr> buffers;
    bool loaded = false;
    bool threw = false;
    try {
        loaded = loader.load_host_data_pipelined(ctx, &buft, buffers, 128, progress, &p);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        if (mode == "invalid") {
            require(message.find("invalid data") != std::string::npos, error.what());
        } else if (mode == "allocation-error") {
            require(message.find("unable to allocate") != std::string::npos, error.what());
        } else if (mode == "read-error") {
            require(message.find("end of file") != std::string::npos, error.what());
        } else {
            throw;
        }
        threw = true;
    }
    if (mode == "invalid" || mode == "allocation-error" || mode == "read-error") {
        require(threw, "expected failure was lost");
        if (mode == "allocation-error") {
            require(p.overlap && loader.size_done == 128, "allocation failure did not join the active reader");
        }
    } else if (mode.rfind("cancel", 0) == 0) {
        require(!loaded, "cancellation was ignored");
        require(!p.progress.empty() && p.progress.back() >= p.cancel_at, "cancellation callback missing");
        if (mode != "cancel-final") {
            require(loader.size_done < loader.size_data, "cancelled load read the entire model");
        }
    } else {
        require(loaded && p.overlap, "pipeline failed");
        require(buffers.size() == 3, "batch grouping or oversized tensor handling failed");
        require(loader.size_done == loader.size_data && p.progress.back() == 1.0f, "incomplete load");
        for (int i = 0; i < 4; ++i) {
            auto * t = ggml_get_tensor(ctx, ("weight_" + std::to_string(i)).c_str());
            require(t && t->buffer, "tensor not allocated");
            for (int j = 0; j < t->ne[0]; ++j) {
                require(static_cast<float *>(t->data)[j] == float(100 * i + j), "loaded bytes differ");
            }
        }
    }
    std::printf("PASS load pipeline: %s\n", mode.c_str());
}

int main() {
    try {
        for (const char * mode : {"success", "cancel-first", "cancel", "cancel-final", "invalid", "read-error", "allocation-error"}) {
            run(mode);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return 0;
}

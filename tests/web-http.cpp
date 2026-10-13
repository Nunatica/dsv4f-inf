#include "common.h"
#include "server-http.h"
#include <cpp-httplib/httplib.h>
#include <cstdio>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void check(const char * web, const std::string & prefix, bool ui) {
    common_params params;
    params.hostnames = {"127.0.0.1"};
    params.port = 0;
    params.n_threads_http = 1;
    params.ui = ui;
    params.public_path = web;
    params.api_prefix = prefix;
    params.api_keys = {"test-key"};
    server_http_context server;
    require(server.init(params), "HTTP init failed");
    auto response = [](const server_http_req &) {
        auto result = std::make_unique<server_http_res>();
        result->data = "{\"ok\":true}";
        return result;
    };
    server.get("/props", response);
    server.post("/v1/chat/completions", response);
    require(server.start(), "HTTP start failed");
    try {
        httplib::Client client("127.0.0.1", server.port);
        client.set_read_timeout(5);
        // UI assets must be available while loading and after ready, without a key.
        for (bool ready : {false, true}) {
            server.is_ready = ready;
            for (const char * path : {"/", "/index.html", "/app.css", "/app.js", "/metrics.js", "/settings.js", "/benchmark.json"}) {
                const auto r = client.Get(prefix + path);
                require(r && r->status == (ui ? 200 : ready ? 401 : 503), "asset auth/loading regression");
                if (ui) {
                    require(!r->body.empty(), "empty asset");
                    if (std::string(path) == "/app.css") {
                        require(r->get_header_value("Content-Type").find("text/css") == 0, "wrong CSS MIME type");
                        require(r->body.find("#sidebar{position:fixed") != std::string::npos, "wrong CSS body");
                    }
                }
            }
        }
        auto css = client.Head(prefix + "/app.css?v=2");
        require(css && css->status == (ui ? 200 : 401), "HEAD/query asset regression");
        auto props = client.Get(prefix + "/props");
        require(props && props->status == 401, "props became public");
        auto unknown = client.Get(prefix + "/private.css");
        require(unknown && unknown->status == 401, "arbitrary CSS became public");
        auto completion = client.Post(prefix + "/v1/chat/completions", "{}", "application/json");
        require(completion && completion->status == 401, "inference became public");
        const httplib::Headers good = {{"Authorization", "Bearer test-key"}};
        const httplib::Headers bad = {{"Authorization", "Bearer wrong-key"}};
        completion = client.Post(prefix + "/v1/chat/completions", bad, "{}", "application/json");
        require(completion && completion->status == 401, "wrong API key accepted");
        completion = client.Post(prefix + "/v1/chat/completions", good, "{}", "application/json");
        require(completion && completion->status == 200, "valid API key rejected");
        auto post_asset = client.Post(prefix + "/app.css", "{}", "application/json");
        require(post_asset && post_asset->status == 401, "asset POST bypassed auth");
        server.stop();
        server.join();
    } catch (...) {
        server.stop();
        server.join();
        throw;
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "web directory required");
        check(argv[1], "", true);
        check(argv[1], "/ui", true);
        check(argv[1], "", false);
        std::puts("PASS: static UI assets load without a key; inference keeps API-key authentication");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}

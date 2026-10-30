#pragma once
#include <string>
#include <cstdint>
#include <utility>
#include <vector>

// Initialize before loading weights. start() seeds initialized GPU caches from
// loaded history at the quiescent startup boundary, then enables counting.
// Call only with the previous model and its services unloaded.
void dsv4f_routing_init(const std::string & model_path);
void dsv4f_routing_start();
// Drains decode telemetry and atomically replaces models/<model stem>.routing.json.
// Errors are logged; a completed response must still reach the client.
bool dsv4f_routing_save();
struct dsv4f_pruning_group {
    std::string gate_name;
    std::vector<int32_t> local; // global-to-format-local expert IDs
    uint64_t expert_bytes; // gate + up + down for one expert, without padding
};
struct dsv4f_pruning_layer {
    int layer, n_experts, n_used;
    std::vector<dsv4f_pruning_group> groups;
};
// Plan all layers together before changing tensor shapes. Empty when disabled;
// otherwise one 0 / -infinity mask per input layer, in global expert ID order.
std::vector<std::vector<float>> dsv4f_pruning_masks(const std::vector<dsv4f_pruning_layer> & layers);

// Repair only excluded hash routes, preserving surviving IDs and top-k uniqueness.
void dsv4f_pruning_repair_hash(std::vector<int32_t> & ids, int top_k,
    const std::string & gate_name, const std::vector<float> & mask);

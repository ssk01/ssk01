#pragma once
#include <vector>
#include "../graph/graph.h"

namespace lf {

// 设备放置 —— 对应 commit 1 的 simple_placer.cc:
//   显式 device 指定优先; 未指定 → FilterSupportedDevices 按 kernel 注册表筛出
//   该 op 支持哪些设备, 再按优先级选 (Order(GPU)=2 < Order(CPU)=3 → 有 GPU
//   kernel 就优先 GPU)。soft placement 在我们 demo 里表现为: 无 GPU kernel 的
//   op 永远只出现在 CPU 候选里 (见 HasMetalKernel)。
inline bool HasMetalKernel(NodeType t) { return t == MATMUL; }  // kernel 注册表 (GPU 侧)

inline std::vector<Device> SimplePlace(const Graph& graph, bool gpu_available) {
    std::vector<Device> devs(graph.id_count(), Device::CPU);
    for (const auto& un : graph.nodes()) {
        Node* n = un.get();
        if (n->device != Device::AUTO) {  // 显式指定优先 (AUTO = 未指定)
            devs[n->id] = n->device;
            continue;
        }
        devs[n->id] = (gpu_available && HasMetalKernel(n->type)) ? Device::GPU
                                                                  : Device::CPU;
    }
    return devs;
}

}  // namespace lf

#pragma once
#include <vector>
#include "../graph/graph.h"

namespace lf {

// 设备放置 —— 对应 commit 1 的 simple_placer.cc:
//   显式 device 指定优先; 未指定 → FilterSupportedDevices 按 kernel 注册表筛出
//   该 op 支持哪些设备, 再按优先级选 (Order(GPU)=2 < Order(CPU)=3 → 有 GPU
//   kernel 就优先 GPU)。soft placement 在我们 demo 里表现为: 无 GPU kernel 的
//   op 永远只出现在 CPU 候选里 (见 HasMetalKernel)。
//
//   colocation: 某些 op 必须和它的输入在同一设备 (TF 的 colocate_with 机制):
//   - SGD_STEP 必须和它更新的 VARIABLE 在同一设备 (直接写变量的状态)
//   - SEND/RECV 已经在图分区时处理, 这里不管
inline bool HasMetalKernel(NodeType t) { return t == MATMUL; }  // kernel 注册表 (GPU 侧)

inline std::vector<Device> SimplePlace(const Graph& graph, bool gpu_available) {
    std::vector<Device> devs(graph.id_count(), Device::CPU);

    // 第一遍: 处理显式指定和普通节点
    for (const auto& un : graph.nodes()) {
        Node* n = un.get();
        if (n->device != Device::AUTO) {  // 显式指定优先 (AUTO = 未指定)
            devs[n->id] = n->device;
            continue;
        }
        devs[n->id] = (gpu_available && HasMetalKernel(n->type)) ? Device::GPU
                                                                  : Device::CPU;
    }

    // 第二遍: 处理 colocation 约束
    // SGD_STEP(var, grad): 必须和 var 在同一设备 (对应 TF 的 colocate_with)
    for (const auto& un : graph.nodes()) {
        Node* n = un.get();
        if (n->type == SGD_STEP && !n->inputs.empty()) {
            Node* var = n->inputs[0];  // SGD_STEP 的第一个输入是变量
            devs[n->id] = devs[var->id];  // 强制和变量在同一设备
        }
    }

    return devs;
}

}  // namespace lf

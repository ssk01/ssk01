#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include "graph.h"

namespace lf {

// 图分区 (v8) — 对应 TF commit 1 的 graph_partition.cc (1050 行)
// 输入: 完整图 + 每个节点的设备分配 (v7 SimplePlace 的产物)
// 输出: 修改后的图 (跨设备边处插入 Send/Recv 节点)
//
// 核心逻辑:
//   for each edge (src -> dst):
//     if src.device != dst.device:
//       在 src 后插入 Send 节点
//       在 dst 前插入 Recv 节点
//       两者通过 rendezvous_key 配对
//
// rendezvous_key 格式: "src_name:output_idx:dst_name"
// (TF 用 tensor_name attr, 格式略有不同, 但思想相同)

inline std::string MakeRendezvousKey(const std::string& src_name, int output_idx,
                                      const std::string& dst_name) {
    return src_name + ":" + std::to_string(output_idx) + ":" + dst_name;
}

// 图分区主函数
// devices: SimplePlace 的输出 (每个节点的设备分配)
// 返回: 是否插入了 Send/Recv (用于判断是否需要 rendezvous)
inline bool PartitionGraph(Graph& graph, const std::vector<Device>& devices) {
    bool has_cross_device_edge = false;

    // 记录每个节点的消费者 (dst) 列表
    std::unordered_map<Node*, std::vector<Node*>> consumers;
    for (const auto& un : graph.nodes()) {
        Node* n = un.get();
        for (Node* input : n->inputs) {
            consumers[input].push_back(n);
        }
    }

    // DupRecvTable: 同一对 (src, output_idx, dst_device) 只创建一个 Recv
    // key = "src_name:output_idx:dst_device"
    // value = Recv 节点指针
    std::unordered_map<std::string, Node*> dup_recv_table;

    // 遍历所有边, 找跨设备边
    std::vector<Node*> nodes_snapshot;
    for (const auto& un : graph.nodes()) {
        nodes_snapshot.push_back(un.get());
    }

    for (Node* src : nodes_snapshot) {
        if (!consumers.count(src)) continue;

        Device src_device = devices[src->id];

        // 对 src 的每个消费者
        for (Node* dst : consumers[src]) {
            Device dst_device = devices[dst->id];

            if (src_device == dst_device) {
                continue;  // 同设备, 无需 Send/Recv
            }

            has_cross_device_edge = true;

            // 跨设备边: src (src_device) -> dst (dst_device)
            // 插入: src -> Send (src_device) ... Recv (dst_device) -> dst

            std::string rendezvous_key = MakeRendezvousKey(src->name, 0, dst->name);

            // 1. 创建 Send 节点 (在 src_device 上)
            Node* send = graph.send("send_" + rendezvous_key, src);
            send->device = src_device;
            send->rendezvous_key = rendezvous_key;

            // 2. 创建或复用 Recv 节点 (在 dst_device 上)
            // DupRecvTable: 如果多个 dst 都消费同一个 src, 只创建一个 Recv
            std::string dup_key = src->name + ":0:" +
                                  (dst_device == Device::CPU ? "CPU" : "GPU");
            Node* recv = nullptr;
            auto it = dup_recv_table.find(dup_key);
            if (it != dup_recv_table.end()) {
                recv = it->second;
            } else {
                recv = graph.recv("recv_" + rendezvous_key);
                recv->device = dst_device;
                recv->rendezvous_key = rendezvous_key;
                // 关键修复: Recv 依赖 Send, 确保拓扑序中 Send 在 Recv 之前执行
                // 虽然数据通过 rendezvous 传递, 但依赖关系避免死锁
                recv->inputs.push_back(send);
                dup_recv_table[dup_key] = recv;
            }

            // 3. 重定向 dst 的输入: src -> recv
            for (size_t i = 0; i < dst->inputs.size(); ++i) {
                if (dst->inputs[i] == src) {
                    dst->inputs[i] = recv;
                }
            }
        }
    }

    return has_cross_device_edge;
}

}  // namespace lf

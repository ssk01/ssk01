#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "../graph/graph.h"
#include "../runtime.h"

namespace lf {

// OpKernelContext: kernel 执行时的上下文 (统一接口)
struct OpKernelContext {
    const Node* node;
    RunState* state;

    // 获取输入 tensor
    const Tensor& input(int i) const {
        return state->out(node->inputs[i]);
    }

    // 设置输出 tensor
    void set_output(const Tensor& t) {
        state->out(node) = t;
    }

    // 获取属性 (简化版: 只支持 float scalar)
    float get_attr_float(const std::string& name) const {
        return node->scalar;  // 简化: 假设 scalar 字段存属性
    }
};

// KernelFunc: kernel 实现函数
using KernelFunc = std::function<void(OpKernelContext*)>;

// KernelDef: kernel 的定义
struct KernelDef {
    std::string op_name;
    Device device;
    KernelFunc kernel_fn;

    KernelDef() : device(Device::CPU) {}  // 默认构造函数

    KernelDef(const std::string& op, Device dev, KernelFunc fn)
        : op_name(op), device(dev), kernel_fn(std::move(fn)) {}
};

// KernelDefBuilder: 链式构建 KernelDef
class KernelDefBuilder {
public:
    explicit KernelDefBuilder(const std::string& op_name) : op_name_(op_name) {}

    KernelDefBuilder& DeviceType(lf::Device dev) {
        device_ = dev;
        return *this;
    }

    KernelDefBuilder& Kernel(KernelFunc fn) {
        kernel_fn_ = std::move(fn);
        return *this;
    }

    void Finalize();

private:
    std::string op_name_;
    enum Device device_ = Device::CPU;
    KernelFunc kernel_fn_;
};

// KernelRegistry: 全局 kernel 注册表
class KernelRegistry {
public:
    static KernelRegistry& Global() {
        static KernelRegistry instance;
        return instance;
    }

    void Register(const KernelDef& def) {
        std::string key = MakeKey(def.op_name, def.device);
        registry_[key] = def;
    }

    const KernelDef* LookUp(const std::string& op_name, Device device) const {
        std::string key = MakeKey(op_name, device);
        auto it = registry_.find(key);
        return (it != registry_.end()) ? &it->second : nullptr;
    }

    // 查找 op 支持的设备列表
    std::vector<Device> GetSupportedDevices(const std::string& op_name) const {
        std::vector<Device> devices;
        for (const auto& [key, def] : registry_) {
            if (def.op_name == op_name) {
                devices.push_back(def.device);
            }
        }
        return devices;
    }

private:
    KernelRegistry() = default;

    static std::string MakeKey(const std::string& op_name, Device device) {
        std::string dev_str = (device == Device::GPU) ? "GPU" : "CPU";
        return op_name + ":" + dev_str;
    }

    std::unordered_map<std::string, KernelDef> registry_;
};

inline void KernelDefBuilder::Finalize() {
    KernelRegistry::Global().Register(KernelDef(op_name_, device_, kernel_fn_));
}

// 宏简化注册
#define REGISTER_KERNEL(op_name, device) \
    static KernelDefBuilder __kernel_register_##op_name##_##device##__ __attribute__((unused)) = \
        KernelDefBuilder(#op_name).Device(Device::device)

}  // namespace lf

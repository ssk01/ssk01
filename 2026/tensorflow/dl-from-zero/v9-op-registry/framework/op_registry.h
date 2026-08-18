#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace lf {

// ArgDef: op 的输入/输出定义
struct ArgDef {
    std::string name;
    std::string type;  // "float", "int32", "T" (模板类型)

    ArgDef(const std::string& n, const std::string& t) : name(n), type(t) {}
};

// AttrDef: op 的属性定义
struct AttrDef {
    std::string name;
    std::string type;      // "bool", "int", "float", "string"
    std::string default_value;

    AttrDef(const std::string& n, const std::string& t, const std::string& dv = "")
        : name(n), type(t), default_value(dv) {}
};

// OpDef: op 的完整定义 (对应 TF OpDef proto)
struct OpDef {
    std::string name;
    std::vector<ArgDef> inputs;
    std::vector<ArgDef> outputs;
    std::vector<AttrDef> attrs;
    bool is_stateful = false;  // 有状态 op 不能被 CSE 合并

    OpDef() = default;
    explicit OpDef(const std::string& n) : name(n) {}
};

// OpDefBuilder: 链式构建 OpDef (对应 TF OpDefBuilder)
class OpDefBuilder {
public:
    explicit OpDefBuilder(const std::string& name) : def_(name) {}

    OpDefBuilder& Input(const std::string& spec) {
        // spec 格式: "name: type" 例如 "a: float"
        auto pos = spec.find(':');
        if (pos != std::string::npos) {
            std::string name = spec.substr(0, pos);
            std::string type = spec.substr(pos + 1);
            // 去除空格
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            type.erase(0, type.find_first_not_of(" \t"));
            type.erase(type.find_last_not_of(" \t") + 1);
            def_.inputs.emplace_back(name, type);
        }
        return *this;
    }

    OpDefBuilder& Output(const std::string& spec) {
        auto pos = spec.find(':');
        if (pos != std::string::npos) {
            std::string name = spec.substr(0, pos);
            std::string type = spec.substr(pos + 1);
            name.erase(0, name.find_first_not_of(" \t"));
            name.erase(name.find_last_not_of(" \t") + 1);
            type.erase(0, type.find_first_not_of(" \t"));
            type.erase(type.find_last_not_of(" \t") + 1);
            def_.outputs.emplace_back(name, type);
        }
        return *this;
    }

    OpDefBuilder& Attr(const std::string& spec) {
        // spec 格式: "name: type = default" 例如 "transpose_a: bool = false"
        auto colon = spec.find(':');
        auto equal = spec.find('=');

        std::string name = spec.substr(0, colon);
        std::string type = (equal != std::string::npos)
                          ? spec.substr(colon + 1, equal - colon - 1)
                          : spec.substr(colon + 1);
        std::string default_val = (equal != std::string::npos)
                                 ? spec.substr(equal + 1)
                                 : "";

        // 去除空格
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t") + 1);
        type.erase(0, type.find_first_not_of(" \t"));
        type.erase(type.find_last_not_of(" \t") + 1);
        if (!default_val.empty()) {
            default_val.erase(0, default_val.find_first_not_of(" \t"));
            default_val.erase(default_val.find_last_not_of(" \t") + 1);
        }

        def_.attrs.emplace_back(name, type, default_val);
        return *this;
    }

    OpDefBuilder& SetIsStateful() {
        def_.is_stateful = true;
        return *this;
    }

    // 注册到全局注册表
    void Finalize();

    const OpDef& Build() const { return def_; }

private:
    OpDef def_;
};

// OpRegistry: 全局 op 注册表 (单例)
class OpRegistry {
public:
    static OpRegistry& Global() {
        static OpRegistry instance;
        return instance;
    }

    void Register(const OpDef& def) {
        registry_[def.name] = def;
    }

    const OpDef* LookUp(const std::string& name) const {
        auto it = registry_.find(name);
        return (it != registry_.end()) ? &it->second : nullptr;
    }

    // 列出所有注册的 op
    std::vector<std::string> ListOps() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : registry_) {
            names.push_back(name);
        }
        return names;
    }

private:
    OpRegistry() = default;
    std::unordered_map<std::string, OpDef> registry_;
};

inline void OpDefBuilder::Finalize() {
    OpRegistry::Global().Register(def_);
}

// 宏简化注册 (对应 TF 的 REGISTER_OP 宏)
#define REGISTER_OP(name) \
    static OpDefBuilder __op_register_##name##__ __attribute__((unused)) = \
        OpDefBuilder(#name)

}  // namespace lf

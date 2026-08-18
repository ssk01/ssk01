// Python binding for dl-from-zero-4-grad (pybind11)
// 梯度 = 图里的子图 (build_gradients); sess.run(fetches) 返回张量列表
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include "framework/tensor.h"
#include "graph/gradients.h"
#include "graph/graph.h"
#include "graph/prune.h"
#include "kernels/kernels.h"
#include "session.h"

namespace py = pybind11;
using namespace lf;

namespace {

Tensor cast_feed_value(const py::handle& v) {
    if (py::isinstance<Tensor>(v)) {
        return py::cast<Tensor>(v);
    }
    if (py::isinstance<py::list>(v) || py::isinstance<py::tuple>(v)) {
        auto data = py::cast<std::vector<float>>(v);
        return Tensor(data, {static_cast<int>(data.size())});
    }
    return Tensor(py::cast<float>(v));
}

std::vector<Node*> cast_node_list(const py::list& l) {
    std::vector<Node*> out;
    for (auto o : l) out.push_back(py::cast<Node*>(o));
    return out;
}

}  // namespace

PYBIND11_MODULE(lfpy, m) {
    m.doc() = "dl-from-zero-4-grad python binding";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<float>())
        .def(py::init<const std::vector<int>&>())
        .def(py::init<const std::vector<float>&, const std::vector<int>&>())
        .def_property_readonly("data", [](const Tensor& t) { return t.data; })
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape; })
        .def("__repr__", [](const Tensor& t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape.size(); i++) {
                if (i) s += ",";
                s += std::to_string(t.shape[i]);
            }
            s += "], data=[";
            for (size_t i = 0; i < t.data.size() && i < 8; i++) {
                if (i) s += ",";
                s += std::to_string(t.data[i]);
            }
            if (t.data.size() > 8) s += ",...";
            s += "])";
            return s;
        });

    // 图节点: 纯静态, 只有 name/type
    py::class_<Node>(m, "Node")
        .def_property_readonly("name", [](const Node& n) { return n.name; })
        .def_property_readonly("type", [](const Node& n) { return static_cast<int>(n.type); });

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("placeholder", [](Graph& g, const std::string& name,
                               const std::vector<int>& shape) { return g.placeholder(name, shape); },
             py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("variable", [](Graph& g, const std::string& name, float init) { return g.variable(name, init); },
             py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("variable_vec", [](Graph& g, const std::string& name, int n, float init) { return g.variable_vec(name, n, init); },
             py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("add", [](Graph& g, Node* a, Node* b) { return g.add(a, b); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("mul", [](Graph& g, Node* a, Node* b) { return g.mul(a, b); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("sub", [](Graph& g, Node* a, Node* b) { return g.sub(a, b); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("square", [](Graph& g, Node* a) { return g.square(a); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("mean", [](Graph& g, Node* a) { return g.mean(a); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("matmul", [](Graph& g, Node* a, Node* b) { return g.matmul(a, b); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("sigmoid", [](Graph& g, Node* a) { return g.sigmoid(a); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("log", [](Graph& g, Node* a) { return g.log(a); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("recip", [](Graph& g, Node* a) { return g.recip(a); }, py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def("sgd_step", [](Graph& g, Node* var, Node* grad, float lr) { return g.sgd_step(var, grad, lr); },
             py::keep_alive<0, 1>(), py::return_value_policy::reference)
        .def_property_readonly("num_nodes", [](const Graph& g) { return g.nodes().size(); })
        .def_property_readonly("node_names", [](const Graph& g) {
            std::vector<std::string> names;
            for (auto& n : g.nodes()) names.push_back(n->name);
            return names;
        });

    py::class_<Session>(m, "Session")
        .def(py::init<>())
        .def("run", [](Session& s, Graph& g, const py::list& targets,
                       const py::dict& feeds) -> py::list {
            std::vector<Node*> t = cast_node_list(targets);
            std::unordered_map<const Node*, Tensor> feed_map;
            for (auto item : feeds) {
                feed_map[py::cast<Node*>(item.first)] = cast_feed_value(item.second);
            }
            auto results = s.run(g, t, feed_map);
            py::list out;
            for (auto& r : results) out.append(r);
            return out;
        }, py::arg("graph"), py::arg("targets"), py::arg("feeds"))
        .def("assign", [](Session& s, Node* var, py::object v) { s.assign(var, cast_feed_value(v)); },
             py::arg("var"), py::arg("value"))
        .def("var_value", [](Session& s, Node* var) { return s.var_value(var); }, py::arg("var"));

    // 构建独立的梯度子图, 返回 {forward 节点: 梯度节点} (对应 TF gradients.py)
    m.def("build_gradients", [](Graph& g, Node* loss) {
        auto grads = lf::build_gradients(g, loss);
        py::dict d;
        for (auto& [n, gn] : grads) d[py::cast(n)] = py::cast(gn);
        return d;
    }, py::arg("graph"), py::arg("loss"));

    m.def("prune", [](Graph& g, const py::list& targets) {
        lf::prune(g, cast_node_list(targets));
    }, py::arg("graph"), py::arg("targets"));
}

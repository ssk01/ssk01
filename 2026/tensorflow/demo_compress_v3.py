"""
V3: 自动边界检测 + pb 改写
=============================
自动分析计算图：从 user/item 输入出发标记每个节点的归属 (user/item/boundary)，
在 boundary 节点处自动插入 broadcast 子图。不需要指定 user_tower/item_tower。

运行:
    python demo_compress_v3.py
"""

import os
import shutil
import copy
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import node_def_pb2, types_pb2
from collections import deque, defaultdict

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
NAIVE_DIR = os.path.join(WORK_DIR, "saved_models", "compress_naive")
COMPRESSED_V3_DIR = os.path.join(WORK_DIR, "saved_models", "compress_v3")

BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
BENCH_WARMUP = 50
BENCH_ITERS = 200


def const_node(name, dtype, value):
    n = node_def_pb2.NodeDef()
    n.name = name; n.op = "Const"
    n.attr["dtype"].type = dtype
    n.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(value, dtype=dtype))
    return n


def find_model_function(meta_graph):
    for f in meta_graph.graph_def.library.function:
        names = [i.name for i in f.signature.input_arg]
        if 'user_feat' in names and 'item_feat' in names and len(f.node_def) > 10:
            return f
    raise RuntimeError("找不到模型函数")


class GraphAnalyzer:
    """分析 FunctionDef 内的 producer/consumer 关系，标记 user/item 归属"""

    def __init__(self, func_def):
        self.func = func_def
        self.nodes = {n.name: n for n in func_def.node_def}
        self.producers = defaultdict(list)     # tensor → producing node
        self.consumers = defaultdict(list)     # node → list of (input_idx, tensor_name)
        self.attrs = {}                        # node → "user" / "item"

        self._build_graph()

    def _build_graph(self):
        for node in self.func.node_def:
            for idx, inp in enumerate(node.input):
                tensor_name = inp.split(":")[0] if ":" in inp else inp
                tensor_name = tensor_name.lstrip("^")
                self.consumers[node.name].append((idx, inp, tensor_name))
                self.producers[tensor_name].append(node.name)

    def propagate(self, user_inputs, item_inputs):
        """从 user/item 输入出发，BFS 标记下游节点归属"""
        queue = deque()
        for name in user_inputs:
            self.attrs[name] = "user"
            queue.append(name)
        for name in item_inputs:
            self.attrs[name] = "item"
            queue.append(name)

        while queue:
            tensor_name = queue.popleft()
            owner = self.attrs.get(tensor_name, "?")
            for consumer_node in self.producers.get(tensor_name, []):
                node = self.nodes.get(consumer_node)
                if not node:
                    continue
                current = self.attrs.get(consumer_node)
                if current is None:
                    self.attrs[consumer_node] = owner
                    queue.append(consumer_node)
                elif current != owner:
                    self.attrs[consumer_node] = "boundary"

    def find_boundary_inputs(self):
        """返回需要插 broadcast 的 boundary 节点及其 user 输入"""
        result = []
        for node in self.func.node_def:
            if self.attrs.get(node.name) == "boundary":
                user_inps = []
                for idx, inp_raw, tn in self.consumers.get(node.name, []):
                    if inp_raw.startswith("^"):
                        continue
                    inp_node_name = tn
                    if inp_node_name.endswith("/axis"):
                        continue
                    if self.attrs.get(inp_node_name) == "user":
                        user_inps.append((idx, inp_raw))
                if user_inps:
                    result.append((node, user_inps))
        return result


def patch_function(func_def):
    func_def = copy.deepcopy(func_def)

    analyzer = GraphAnalyzer(func_def)
    user_input_names = ["user_feat"]
    item_input_names = ["item_feat"]
    analyzer.propagate(user_input_names, item_input_names)

    boundaries = analyzer.find_boundary_inputs()

    node_list = list(func_def.node_def)
    insertion_offset = 0

    for node, user_inps in boundaries:
        node_idx = next(i for i, n in enumerate(node_list) if n.name == node.name)

        for inp_idx, inp_tensor in user_inps:
            prefix = f"compress_v3/{node.name}/inp{inp_idx}"

            # 需要知道 user tensor 的最后一维大小
            inp_node_name = inp_tensor.split(":")[0]
            inp_node = next((n for n in node_list if n.name == inp_node_name), None)
            if inp_node and '_output_shapes' in inp_node.attr:
                shapes = inp_node.attr['_output_shapes'].list.shape
                if shapes and len(shapes[0].dim) >= 2:
                    last_dim = shapes[0].dim[-1].size
                else:
                    last_dim = 1
            else:
                last_dim = 1

            new_nodes = []

            # Shape(item_emb) 从 node 的另一个非 user 输入获取
            other_inp = None
            for ci, inp_raw, tn in analyzer.consumers.get(node.name, []):
                if ci != inp_idx and not inp_raw.startswith("^"):
                    other_node_name = tn
                    if other_node_name.endswith("/axis"):
                        continue
                    if analyzer.attrs.get(other_node_name) == "item":
                        other_inp = inp_raw
                        break
            if other_inp is None:
                for ii, ir, _ in analyzer.consumers.get(node.name, []):
                    if ii != inp_idx and not ir.startswith("^"):
                        other_inp = ir
                        break

            if other_inp is None:
                continue

            n = node_def_pb2.NodeDef()
            n.name = f"{prefix}/ref_shape"; n.op = "Shape"; n.input.append(other_inp)
            n.attr["T"].type = types_pb2.DT_FLOAT; n.attr["out_type"].type = types_pb2.DT_INT32
            new_nodes.append(n)

            new_nodes.append(const_node(f"{prefix}/begin", types_pb2.DT_INT32, [0]))
            new_nodes.append(const_node(f"{prefix}/end", types_pb2.DT_INT32, [1]))
            new_nodes.append(const_node(f"{prefix}/strides", types_pb2.DT_INT32, [1]))

            n = node_def_pb2.NodeDef()
            n.name = f"{prefix}/batch_scalar"; n.op = "StridedSlice"
            n.input.extend([f"{prefix}/ref_shape:output:0", f"{prefix}/begin:output:0",
                             f"{prefix}/end:output:0", f"{prefix}/strides:output:0"])
            n.attr["T"].type = types_pb2.DT_INT32; n.attr["Index"].type = types_pb2.DT_INT32
            n.attr["begin_mask"].i = 0; n.attr["end_mask"].i = 0; n.attr["shrink_axis_mask"].i = 1
            new_nodes.append(n)

            new_nodes.append(const_node(f"{prefix}/emb_scalar", types_pb2.DT_INT32, last_dim))

            n = node_def_pb2.NodeDef()
            n.name = f"{prefix}/target_shape"; n.op = "Pack"
            n.attr["N"].i = 2; n.attr["T"].type = types_pb2.DT_INT32; n.attr["axis"].i = 0
            n.input.extend([f"{prefix}/batch_scalar:output:0", f"{prefix}/emb_scalar:output:0"])
            new_nodes.append(n)

            n = node_def_pb2.NodeDef()
            n.name = f"{prefix}/broadcast"; n.op = "BroadcastTo"
            n.input.extend([inp_tensor, f"{prefix}/target_shape:output:0"])
            n.attr["T"].type = types_pb2.DT_FLOAT; n.attr["Tidx"].type = types_pb2.DT_INT32
            new_nodes.append(n)

            bc_output = f"{prefix}/broadcast:output:0"

            # 修改 node 的 input
            new_inps = list(node.input)
            new_inps[inp_idx] = bc_output
            del node.input[:]
            node.input.extend(new_inps)

            # 插入新节点
            for nn in reversed(new_nodes):
                node_list.insert(node_idx + insertion_offset, nn)
                insertion_offset += 1

    del func_def.node_def[:]
    func_def.node_def.extend(node_list)
    return func_def


def compress(naive_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(naive_dir, output_dir)

    pb_path = os.path.join(output_dir, "saved_model.pb")
    sm = saved_model_pb2.SavedModel()
    with open(pb_path, "rb") as f:
        sm.ParseFromString(f.read())

    original = find_model_function(sm.meta_graphs[0])
    patched = patch_function(original)

    lib = sm.meta_graphs[0].graph_def.library
    for i, f in enumerate(lib.function):
        if f.signature.name == original.signature.name:
            del lib.function[i]
            lib.function.insert(i, patched)
            break

    with open(pb_path, "wb") as f:
        f.write(sm.SerializeToString())
    return output_dir


def bench(model_dir, N, is_compressed):
    fn = tf.saved_model.load(model_dir).signatures["serving_default"]
    u1 = np.random.randn(1, 3).astype(np.float32)
    items = np.random.randn(N, 3).astype(np.float32)
    ut = tf.constant(u1 if is_compressed else np.tile(u1, (N, 1)))
    it = tf.constant(items)
    for _ in range(BENCH_WARMUP):
        fn(user_feat=ut, item_feat=it)
    lats = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        fn(user_feat=ut, item_feat=it)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    return lats.mean(), np.percentile(lats, 99)


def main():
    print("=" * 60)
    print("V3: DeepRec-style — 自动标记 user/item 归属，找边界插 broadcast")
    print("=" * 60)

    compress(NAIVE_DIR, COMPRESSED_V3_DIR)
    print(f"改写完成: {COMPRESSED_V3_DIR}")

    print("\n数值对拍:")
    n_fn = tf.saved_model.load(NAIVE_DIR).signatures["serving_default"]
    c_fn = tf.saved_model.load(COMPRESSED_V3_DIR).signatures["serving_default"]
    for N in [5, 10, 50, 100]:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        sn = n_fn(user_feat=tf.constant(np.tile(u1, (N, 1))), item_feat=tf.constant(items))
        sc = c_fn(user_feat=tf.constant(u1), item_feat=tf.constant(items))
        diff = np.abs(sn["output_0"].numpy() - sc["output_0"].numpy()).max()
        print(f"  N={N:>4d}  diff={diff:.2e}  {'✓' if diff < 1e-4 else '✗'}")

    print(f"\n{'N':>6} | {'Naive(ms)':>10} {'P99':>8} | {'V3(ms)':>10} {'P99':>8} | {'Speedup':>7}")
    print("-" * 65)
    for N in BENCH_N_LIST:
        nm, np99 = bench(NAIVE_DIR, N, False)
        cm, cp99 = bench(COMPRESSED_V3_DIR, N, True)
        print(f'{N:>6} | {nm:>10.3f} {np99:>8.3f} | {cm:>10.3f} {cp99:>8.3f} | {nm/cm:>6.2f}x')


if __name__ == "__main__":
    main()

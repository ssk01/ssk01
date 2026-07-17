"""
V2: SavedModel pb 改写实现
===========================
直接修改导出的 saved_model.pb，在 user tower 输出和 concat 之间插入 broadcast 子图。
不依赖模型源码，只需已有 saved_model。

运行:
    python demo_compress_v2.py
"""

import os
import shutil
import copy
import time
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.framework import node_def_pb2, types_pb2

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
NAIVE_DIR = os.path.join(WORK_DIR, "saved_models", "compress_naive")
COMPRESSED_V2_DIR = os.path.join(WORK_DIR, "saved_models", "compress_v2")

BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
BENCH_WARMUP = 50
BENCH_ITERS = 200


def const_node(name, dtype, value):
    n = node_def_pb2.NodeDef()
    n.name = name
    n.op = "Const"
    n.attr["dtype"].type = dtype
    n.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(value, dtype=dtype))
    return n


def find_model_function(meta_graph):
    for f in meta_graph.graph_def.library.function:
        names = [i.name for i in f.signature.input_arg]
        if 'user_feat' in names and 'item_feat' in names and len(f.node_def) > 10:
            return f
    raise RuntimeError("找不到模型函数")


def find_node(func, name_part, op_type):
    for n in func.node_def:
        if name_part in n.name and n.op == op_type:
            return n
    return None


def get_emb_dim(func):
    """从 user_emb MatMul 的 kernel shape 提取 embedding 维度"""
    matmul = find_node(func, 'user_emb', 'MatMul')
    if matmul:
        for inp_name in matmul.input:
            read_name = inp_name.split(':')[0] if ':' in inp_name else inp_name
            if 'ReadVariableOp' in read_name:
                read_node = find_node(func, read_name, 'ReadVariableOp')
                if read_node and '_output_shapes' in read_node.attr:
                    shapes = read_node.attr['_output_shapes'].list.shape
                    if shapes and len(shapes[0].dim) == 2:
                        return shapes[0].dim[-1].size
    for n in func.node_def:
        if 'user_emb' in n.name and 'BiasAdd' in n.op and '_output_shapes' in n.attr:
            shapes = n.attr['_output_shapes'].list.shape
            if shapes and len(shapes[0].dim) == 2:
                return shapes[0].dim[-1].size
    return 64


def patch_function(func):
    func = copy.deepcopy(func)
    emb_dim = get_emb_dim(func)

    user_emb_node = find_node(func, 'user_emb', 'BiasAdd')
    item_emb_node = find_node(func, 'item_emb', 'BiasAdd')
    concat_node = find_node(func, 'concat', 'ConcatV2')

    assert user_emb_node and item_emb_node and concat_node, "找不到关键节点"

    user_out = f"{user_emb_node.name}:output:0"
    item_out = f"{item_emb_node.name}:output:0"
    prefix = "compress_v2"

    # 节点: Shape(item_emb) → StridedSlice[0] → 与 emb_dim Pack → BroadcastTo(user_emb, ...)
    nodes = []

    # Shape
    n = node_def_pb2.NodeDef()
    n.name = f"{prefix}/item_shape"; n.op = "Shape"; n.input.append(item_out)
    n.attr["T"].type = types_pb2.DT_FLOAT; n.attr["out_type"].type = types_pb2.DT_INT32
    nodes.append(n)

    # Consts for StridedSlice
    for suffix, val in [("/begin", [0]), ("/end", [1]), ("/strides", [1])]:
        nodes.append(const_node(f"{prefix}{suffix}", types_pb2.DT_INT32, val))

    # StridedSlice: 取 Shape 的第0维 → 标量
    n = node_def_pb2.NodeDef()
    n.name = f"{prefix}/batch_scalar"; n.op = "StridedSlice"
    n.input.extend([f"{prefix}/item_shape:output:0",
                     f"{prefix}/begin:output:0", f"{prefix}/end:output:0",
                     f"{prefix}/strides:output:0"])
    n.attr["T"].type = types_pb2.DT_INT32; n.attr["Index"].type = types_pb2.DT_INT32
    n.attr["begin_mask"].i = 0; n.attr["end_mask"].i = 0; n.attr["shrink_axis_mask"].i = 1
    nodes.append(n)

    # emb_dim 常量 (也是标量)
    nodes.append(const_node(f"{prefix}/emb_scalar", types_pb2.DT_INT32, emb_dim))

    # Pack: 把两个标量打包成 [2] 向量 (广播目标 shape)
    n = node_def_pb2.NodeDef()
    n.name = f"{prefix}/target_shape"; n.op = "Pack"; n.attr["N"].i = 2
    n.attr["T"].type = types_pb2.DT_INT32; n.attr["axis"].i = 0
    n.input.extend([f"{prefix}/batch_scalar:output:0", f"{prefix}/emb_scalar:output:0"])
    nodes.append(n)

    # BroadcastTo
    n = node_def_pb2.NodeDef()
    n.name = f"{prefix}/broadcast"; n.op = "BroadcastTo"
    n.input.extend([user_out, f"{prefix}/target_shape:output:0"])
    n.attr["T"].type = types_pb2.DT_FLOAT; n.attr["Tidx"].type = types_pb2.DT_INT32
    nodes.append(n)

    bc_output = f"{prefix}/broadcast:output:0"

    # 修改 concat: user_emb 输入换成 broadcast 输出
    new_inputs = []
    for inp in concat_node.input:
        if user_emb_node.name in inp:
            new_inputs.append(bc_output)
        else:
            new_inputs.append(inp)
    del concat_node.input[:]
    concat_node.input.extend(new_inputs)

    # 插入新节点到 concat 前
    idx = next(i for i, n in enumerate(func.node_def) if n.name == concat_node.name)
    for n in reversed(nodes):
        func.node_def.insert(idx, n)

    return func


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
    print("V2: SavedModel pb 改写 — 不依赖源码，插 broadcast 子图")
    print("=" * 60)

    compress(NAIVE_DIR, COMPRESSED_V2_DIR)
    print(f"改写完成: {COMPRESSED_V2_DIR}")

    print("\n数值对拍:")
    n_fn = tf.saved_model.load(NAIVE_DIR).signatures["serving_default"]
    c_fn = tf.saved_model.load(COMPRESSED_V2_DIR).signatures["serving_default"]
    for N in [5, 10, 50, 100]:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        sn = n_fn(user_feat=tf.constant(np.tile(u1, (N, 1))), item_feat=tf.constant(items))
        sc = c_fn(user_feat=tf.constant(u1), item_feat=tf.constant(items))
        diff = np.abs(sn["output_0"].numpy() - sc["output_0"].numpy()).max()
        print(f"  N={N:>4d}  diff={diff:.2e}  {'✓' if diff < 1e-4 else '✗'}")

    print(f"\n{'N':>6} | {'Naive(ms)':>10} {'P99':>8} | {'V2(ms)':>10} {'P99':>8} | {'Speedup':>7}")
    print("-" * 65)
    for N in BENCH_N_LIST:
        nm, np99 = bench(NAIVE_DIR, N, False)
        cm, cp99 = bench(COMPRESSED_V2_DIR, N, True)
        print(f'{N:>6} | {nm:>10.3f} {np99:>8.3f} | {cm:>10.3f} {cp99:>8.3f} | {nm/cm:>6.2f}x')


if __name__ == "__main__":
    main()

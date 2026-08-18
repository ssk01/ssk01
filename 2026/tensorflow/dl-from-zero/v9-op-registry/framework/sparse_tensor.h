#pragma once
#include <vector>
#include <algorithm>
#include "tensor.h"

namespace lf {

// SparseTensor: 稀疏张量表示 (对应 TF SparseTensor)
// 核心: indices + values + dense_shape
//
// 例子: 稠密张量 [0, 0, 3, 0, 0, 5]
//   → 稀疏: indices = [2, 5]
//           values = [3.0, 5.0]
//           dense_shape = [6]
//
// 搜广推场景: 百万维特征向量只有几十个非零 → 节省 99.99% 内存
struct SparseTensor {
    Tensor indices;      // shape = [nnz] (一维) 或 [nnz, ndims] (多维)
    Tensor values;       // shape = [nnz]
    std::vector<int> dense_shape;  // 对应稠密张量的形状

    SparseTensor() = default;

    SparseTensor(const Tensor& idx, const Tensor& val, const std::vector<int>& shape)
        : indices(idx), values(val), dense_shape(shape) {}

    // 非零元素个数
    int nnz() const {
        return static_cast<int>(values.size());
    }

    // 转为稠密张量 (用于验证/对比)
    Tensor to_dense() const {
        Tensor dense(dense_shape);
        std::fill(dense.data.begin(), dense.data.end(), 0.0f);

        if (dense_shape.size() == 1) {
            // 一维情况
            for (int i = 0; i < nnz(); ++i) {
                int idx = static_cast<int>(indices.data[i]);
                dense.data[idx] = values.data[i];
            }
        } else if (dense_shape.size() == 2) {
            // 二维情况: indices shape = [nnz, 2]
            int cols = dense_shape[1];
            for (int i = 0; i < nnz(); ++i) {
                int row = static_cast<int>(indices.data[i * 2]);
                int col = static_cast<int>(indices.data[i * 2 + 1]);
                dense.data[row * cols + col] = values.data[i];
            }
        }

        return dense;
    }

    // 从稠密张量创建稀疏张量
    static SparseTensor from_dense(const Tensor& dense, float threshold = 1e-6f) {
        std::vector<int> idx_list;
        std::vector<float> val_list;

        if (dense.shape.size() == 1) {
            // 一维
            for (int i = 0; i < dense.size(); ++i) {
                if (std::abs(dense.data[i]) > threshold) {
                    idx_list.push_back(i);
                    val_list.push_back(dense.data[i]);
                }
            }

            Tensor indices(std::vector<int>{static_cast<int>(idx_list.size())});
            for (size_t i = 0; i < idx_list.size(); ++i) {
                indices.data[i] = static_cast<float>(idx_list[i]);
            }

            Tensor values(std::vector<int>{static_cast<int>(val_list.size())});
            values.data = val_list;

            return SparseTensor(indices, values, dense.shape);
        }

        // 多维情况暂不实现
        return SparseTensor();
    }

    // 内存占用 (字节)
    size_t memory_bytes() const {
        return indices.data.size() * sizeof(float) +
               values.data.size() * sizeof(float) +
               dense_shape.size() * sizeof(int);
    }
};

// 稀疏 matmul: SparseTensor[M, K] × DenseTensor[K] → DenseTensor[M]
// 对应 TF SparseMatMul (简化版: 只支持 matvec)
inline Tensor sparse_matmul(const SparseTensor& sparse, const Tensor& dense) {
    // sparse.dense_shape = [M, K], dense.shape = [K] → output.shape = [M]
    int M = sparse.dense_shape[0];

    Tensor output(std::vector<int>{M});
    std::fill(output.data.begin(), output.data.end(), 0.0f);

    // sparse.indices shape = [nnz, 2], sparse.values shape = [nnz]
    for (int i = 0; i < sparse.nnz(); ++i) {
        int row = static_cast<int>(sparse.indices.data[i * 2]);
        int col = static_cast<int>(sparse.indices.data[i * 2 + 1]);
        float val = sparse.values.data[i];
        output.data[row] += val * dense.data[col];
    }

    return output;
}

// 稀疏 reduce_sum: SparseTensor → scalar
inline Tensor sparse_reduce_sum(const SparseTensor& sparse) {
    float sum = 0.0f;
    for (int i = 0; i < sparse.nnz(); ++i) {
        sum += sparse.values.data[i];
    }
    return Tensor(sum);
}

}  // namespace lf

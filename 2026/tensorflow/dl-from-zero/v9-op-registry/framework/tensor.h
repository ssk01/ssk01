#pragma once
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace lf {

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() : data(std::vector<float>{0.0f}), shape(std::vector<int>{}) {}

    explicit Tensor(float val) : data({val}), shape({}) {}

    explicit Tensor(const std::vector<int>& s)
        : data(std::accumulate(s.begin(), s.end(), 1,
                               std::multiplies<int>{}), 0.0f), shape(s) {}

    Tensor(const std::vector<float>& d, const std::vector<int>& s)
        : data(d), shape(s) {}

    int size() const {
        if (shape.empty()) return 1;
        int total = 1;
        for (int d : shape) total *= d;
        return total;
    }

    void reshape(const std::vector<int>& new_shape) {
        shape = new_shape;
    }

    bool is_empty() const {
        return data.empty() || (data.size() == 1 && shape.empty() && data[0] == 0.0f);
    }
};

inline Tensor tensor_add(const Tensor& a, const Tensor& b) {
    Tensor out(a.shape.empty() ? b.shape : a.shape);
    if (a.shape.empty()) {
        float s = a.data[0];
        for (int i = 0; i < b.size(); i++) out.data[i] = s + b.data[i];
    } else if (b.shape.empty()) {
        float s = b.data[0];
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] + s;
    } else {
        assert(a.shape == b.shape);
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] + b.data[i];
    }
    return out;
}

inline Tensor tensor_sub(const Tensor& a, const Tensor& b) {
    Tensor out(a.shape.empty() ? b.shape : a.shape);
    if (a.shape.empty()) {
        float s = a.data[0];
        for (int i = 0; i < b.size(); i++) out.data[i] = s - b.data[i];
    } else if (b.shape.empty()) {
        float s = b.data[0];
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] - s;
    } else {
        assert(a.shape == b.shape);
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] - b.data[i];
    }
    return out;
}

inline Tensor tensor_mul(const Tensor& a, const Tensor& b) {
    Tensor out(a.shape.empty() ? b.shape : a.shape);
    if (a.shape.empty()) {
        float s = a.data[0];
        for (int i = 0; i < b.size(); i++) out.data[i] = s * b.data[i];
    } else if (b.shape.empty()) {
        float s = b.data[0];
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] * s;
    } else {
        assert(a.shape == b.shape);
        for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] * b.data[i];
    }
    return out;
}

inline Tensor tensor_square(const Tensor& a) {
    Tensor out(a.shape);
    for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] * a.data[i];
    return out;
}

inline Tensor tensor_mean(const Tensor& a) {
    float sum = std::accumulate(a.data.begin(), a.data.end(), 0.0f);
    return Tensor(sum / static_cast<float>(a.size()));
}

inline Tensor tensor_mul_scalar(const Tensor& a, float s) {
    Tensor out(a.shape);
    for (int i = 0; i < a.size(); i++) out.data[i] = a.data[i] * s;
    return out;
}

inline float tensor_sum(const Tensor& a) {
    return std::accumulate(a.data.begin(), a.data.end(), 0.0f);
}

// 张量的数值范围 (量化时需要)
inline std::pair<float, float> tensor_minmax(const Tensor& a) {
    float mn = a.data[0], mx = a.data[0];
    for (float v : a.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    return {mn, mx};
}

// a: [N, F], b: [F]  ->  [N]
inline Tensor tensor_matmul(const Tensor& a, const Tensor& b) {
    int N = a.shape[0], F = a.shape[1];
    Tensor out(std::vector<int>{N});
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int f = 0; f < F; f++) acc += a.data[n * F + f] * b.data[f];
        out.data[n] = acc;
    }
    return out;
}

inline Tensor tensor_sigmoid(const Tensor& a) {
    Tensor out(a.shape);
    for (int i = 0; i < a.size(); i++)
        out.data[i] = 1.0f / (1.0f + std::exp(-a.data[i]));
    return out;
}

// 数值保护: 输入 clip 到 >= 1e-7, 避免 log(0)
inline Tensor tensor_log(const Tensor& a) {
    const float eps = 1e-7f;
    Tensor out(a.shape);
    for (int i = 0; i < a.size(); i++)
        out.data[i] = std::log(std::max(a.data[i], eps));
    return out;
}

inline Tensor tensor_recip(const Tensor& a) {
    const float eps = 1e-7f;
    Tensor out(a.shape);
    for (int i = 0; i < a.size(); i++) out.data[i] = 1.0f / std::max(a.data[i], eps);
    return out;
}

// MEAN 的梯度: grad 是标量, 输出填成 x 的形状, 每个元素 = grad / x.size()
// (对应 TF MeanGrad; 因为 shape 要到运行时才确定, 所以用专用 kernel)
inline Tensor tensor_mean_grad(const Tensor& grad, const Tensor& x) {
    Tensor out(x.shape);
    float g = grad.data[0] / static_cast<float>(x.size());
    for (int i = 0; i < x.size(); i++) out.data[i] = g;
    return out;
}

// MATMUL 对 A 的梯度: dA[n,f] = grad[n] * B[f]   (外积, 对应 TF MatMulGrad)
inline Tensor tensor_matmul_grad_a(const Tensor& grad, const Tensor& b) {
    int N = grad.shape[0], F = b.shape[0];
    Tensor out(std::vector<int>{N, F});
    for (int n = 0; n < N; n++)
        for (int f = 0; f < F; f++) out.data[n * F + f] = grad.data[n] * b.data[f];
    return out;
}

// MATMUL 对 B 的梯度: dB[f] = sum_n grad[n] * A[n,f]
inline Tensor tensor_matmul_grad_b(const Tensor& grad, const Tensor& a) {
    int N = a.shape[0], F = a.shape[1];
    Tensor out(std::vector<int>{F});
    for (int f = 0; f < F; f++) {
        float acc = 0.0f;
        for (int n = 0; n < N; n++) acc += grad.data[n] * a.data[n * F + f];
        out.data[f] = acc;
    }
    return out;
}

inline void tensor_add_to(Tensor& a, const Tensor& b) {
    if (a.shape.empty() && b.shape.empty()) {
        a.data[0] += b.data[0];
    } else if (a.shape.empty()) {
        float& v = a.data[0];
        for (int i = 0; i < b.size(); i++) v += b.data[i];
    } else if (b.shape.empty()) {
        float v = b.data[0];
        for (int i = 0; i < a.size(); i++) a.data[i] += v;
    } else {
        assert(a.shape == b.shape);
        for (int i = 0; i < a.size(); i++) a.data[i] += b.data[i];
    }
}

inline Tensor tensor_broadcast_grad(const Tensor& grad, const std::vector<int>& target_shape) {
    if (target_shape.empty()) {
        return Tensor(tensor_sum(grad));
    }
    return grad;
}

// 静态分配的空张量 —— 避免每次调用处都 new 一个空 Tensor
// (对应 Performance Hints 里 empty_device_info() 的 static 共享对象模式)
inline const Tensor& kEmptyTensor() {
    static const Tensor t;
    return t;
}

}  // namespace lf

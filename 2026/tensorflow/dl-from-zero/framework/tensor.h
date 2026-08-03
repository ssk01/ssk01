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

}  // namespace lf

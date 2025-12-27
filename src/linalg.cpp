#include "np/linalg.h"

#include <cstdint>

namespace np {

namespace {

template <typename T>
NDArray* dot_1d_1d(const NDArray& a, const NDArray& b) {
    std::size_t n = a.shape[0];
    std::size_t shape[] = {1};
    NDArray* out = create_array(shape, 1, a.dtype);
    if (!out) {
        return nullptr;
    }
    T sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const T* pa = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + i * a.strides[0]);
        const T* pb = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(b.data) + i * b.strides[0]);
        sum += (*pa) * (*pb);
    }
    *static_cast<T*>(out->data) = sum;
    return out;
}

template <typename T>
NDArray* dot_2d_1d(const NDArray& a, const NDArray& b) {
    std::size_t m = a.shape[0];
    std::size_t k = a.shape[1];
    std::size_t shape[] = {m};
    NDArray* out = create_array(shape, 1, a.dtype);
    if (!out) {
        return nullptr;
    }
    for (std::size_t i = 0; i < m; ++i) {
        T sum = 0;
        for (std::size_t j = 0; j < k; ++j) {
            const T* pa = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + i * a.strides[0] + j * a.strides[1]);
            const T* pb = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(b.data) + j * b.strides[0]);
            sum += (*pa) * (*pb);
        }
        *reinterpret_cast<T*>(static_cast<std::uint8_t*>(out->data) + i * out->strides[0]) = sum;
    }
    return out;
}

template <typename T>
NDArray* dot_1d_2d(const NDArray& a, const NDArray& b) {
    std::size_t k = a.shape[0];
    std::size_t n = b.shape[1];
    std::size_t shape[] = {n};
    NDArray* out = create_array(shape, 1, a.dtype);
    if (!out) {
        return nullptr;
    }
    for (std::size_t j = 0; j < n; ++j) {
        T sum = 0;
        for (std::size_t i = 0; i < k; ++i) {
            const T* pa = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + i * a.strides[0]);
            const T* pb = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(b.data) + i * b.strides[0] + j * b.strides[1]);
            sum += (*pa) * (*pb);
        }
        *reinterpret_cast<T*>(static_cast<std::uint8_t*>(out->data) + j * out->strides[0]) = sum;
    }
    return out;
}

template <typename T>
NDArray* dot_2d_2d(const NDArray& a, const NDArray& b) {
    std::size_t m = a.shape[0];
    std::size_t k = a.shape[1];
    std::size_t n = b.shape[1];
    std::size_t shape[] = {m, n};
    NDArray* out = create_array(shape, 2, a.dtype);
    if (!out) {
        return nullptr;
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            T sum = 0;
            for (std::size_t p = 0; p < k; ++p) {
                const T* pa = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + i * a.strides[0] + p * a.strides[1]);
                const T* pb = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(b.data) + p * b.strides[0] + j * b.strides[1]);
                sum += (*pa) * (*pb);
            }
            *reinterpret_cast<T*>(static_cast<std::uint8_t*>(out->data) + i * out->strides[0] + j * out->strides[1]) = sum;
        }
    }
    return out;
}

}  // namespace

NDArray* dot(const NDArray& a, const NDArray& b) {
    if (a.dtype != b.dtype) {
        return nullptr;
    }
    if (a.ndim == 1 && b.ndim == 1) {
        if (a.shape[0] != b.shape[0]) {
            return nullptr;
        }
        if (a.dtype == DType::Float32) {
            return dot_1d_1d<float>(a, b);
        }
        return dot_1d_1d<double>(a, b);
    }
    if (a.ndim == 2 && b.ndim == 1) {
        if (a.shape[1] != b.shape[0]) {
            return nullptr;
        }
        if (a.dtype == DType::Float32) {
            return dot_2d_1d<float>(a, b);
        }
        return dot_2d_1d<double>(a, b);
    }
    if (a.ndim == 1 && b.ndim == 2) {
        if (a.shape[0] != b.shape[0]) {
            return nullptr;
        }
        if (a.dtype == DType::Float32) {
            return dot_1d_2d<float>(a, b);
        }
        return dot_1d_2d<double>(a, b);
    }
    if (a.ndim == 2 && b.ndim == 2) {
        if (a.shape[1] != b.shape[0]) {
            return nullptr;
        }
        if (a.dtype == DType::Float32) {
            return dot_2d_2d<float>(a, b);
        }
        return dot_2d_2d<double>(a, b);
    }
    return nullptr;
}

}  // namespace np

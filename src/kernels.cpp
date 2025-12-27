#include "np/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "np/broadcast.h"
#include "np/ndarray.h"

namespace np {

namespace {

template <typename T>
void elementwise_kernel(const BroadcastPlan& plan, const NDArray& a, const NDArray& b, NDArray& out,
                        const std::function<T(T, T)>& op) {
    std::size_t ndim = plan.shape.size();
    std::vector<std::size_t> index(ndim, 0);
    std::size_t total = 1;
    for (std::size_t dim : plan.shape) {
        total *= dim;
    }

    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t offset_a = 0;
        std::size_t offset_b = 0;
        std::size_t offset_out = 0;
        for (std::size_t i = 0; i < ndim; ++i) {
            offset_a += index[i] * plan.strides_a[i];
            offset_b += index[i] * plan.strides_b[i];
            offset_out += index[i] * out.strides[i];
        }
        T* out_ptr = reinterpret_cast<T*>(static_cast<std::uint8_t*>(out.data) + offset_out);
        const T* a_ptr = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + offset_a);
        const T* b_ptr = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(b.data) + offset_b);
        *out_ptr = op(*a_ptr, *b_ptr);

        for (std::size_t i = ndim; i-- > 0;) {
            index[i] += 1;
            if (index[i] < plan.shape[i]) {
                break;
            }
            index[i] = 0;
        }
    }
}

template <typename T>
void unary_kernel(const NDArray& a, NDArray& out, const std::function<T(T)>& op) {
    std::size_t total = numel(a);
    if (total == 0) {
        return;
    }
    if (is_contiguous(a) && is_contiguous(out)) {
        const T* in_ptr = static_cast<const T*>(a.data);
        T* out_ptr = static_cast<T*>(out.data);
        for (std::size_t i = 0; i < total; ++i) {
            out_ptr[i] = op(in_ptr[i]);
        }
        return;
    }
    std::vector<std::size_t> index(a.ndim, 0);
    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t offset = 0;
        std::size_t out_offset = 0;
        for (std::size_t i = 0; i < a.ndim; ++i) {
            offset += index[i] * a.strides[i];
            out_offset += index[i] * out.strides[i];
        }
        const T* in_ptr = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(a.data) + offset);
        T* out_ptr = reinterpret_cast<T*>(static_cast<std::uint8_t*>(out.data) + out_offset);
        *out_ptr = op(*in_ptr);
        for (std::size_t i = a.ndim; i-- > 0;) {
            index[i] += 1;
            if (index[i] < a.shape[i]) {
                break;
            }
            index[i] = 0;
        }
    }
}

NDArray* apply_elementwise(const NDArray& a, const NDArray& b, const std::function<double(double, double)>& op_double,
                           const std::function<float(float, float)>& op_float) {
    if (a.dtype != b.dtype) {
        return nullptr;
    }
    BroadcastPlan plan;
    if (!compute_broadcast(a, b, &plan)) {
        return nullptr;
    }
    NDArray* out = create_array(plan.shape.data(), plan.shape.size(), a.dtype);
    if (!out) {
        return nullptr;
    }
    if (a.dtype == DType::Float32) {
        elementwise_kernel<float>(plan, a, b, *out, op_float);
    } else if (a.dtype == DType::Float64) {
        elementwise_kernel<double>(plan, a, b, *out, op_double);
    } else {
        free_array(out);
        return nullptr;
    }
    return out;
}

NDArray* apply_unary(const NDArray& a, const std::function<double(double)>& op_double,
                     const std::function<float(float)>& op_float) {
    NDArray* out = create_array(a.shape.data(), a.ndim, a.dtype);
    if (!out) {
        return nullptr;
    }
    if (a.dtype == DType::Float32) {
        unary_kernel<float>(a, *out, op_float);
    } else if (a.dtype == DType::Float64) {
        unary_kernel<double>(a, *out, op_double);
    } else {
        free_array(out);
        return nullptr;
    }
    return out;
}

}  // namespace

NDArray* add(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return x + y; },
                             [](float x, float y) { return x + y; });
}

NDArray* sub(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return x - y; },
                             [](float x, float y) { return x - y; });
}

NDArray* mul(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return x * y; },
                             [](float x, float y) { return x * y; });
}

NDArray* div(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return x / y; },
                             [](float x, float y) { return x / y; });
}

NDArray* maximum(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return std::max(x, y); },
                             [](float x, float y) { return std::max(x, y); });
}

NDArray* minimum(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return std::min(x, y); },
                             [](float x, float y) { return std::min(x, y); });
}

NDArray* pow(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return std::pow(x, y); },
                             [](float x, float y) { return std::pow(x, y); });
}

NDArray* exp(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::exp(x); }, [](float x) { return std::exp(x); });
}

NDArray* log(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::log(x); }, [](float x) { return std::log(x); });
}

NDArray* sqrt(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::sqrt(x); }, [](float x) { return std::sqrt(x); });
}

NDArray* abs(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::abs(x); }, [](float x) { return std::abs(x); });
}

NDArray* sin(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::sin(x); }, [](float x) { return std::sin(x); });
}

NDArray* cos(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::cos(x); }, [](float x) { return std::cos(x); });
}

NDArray* tan(const NDArray& a) {
    return apply_unary(a, [](double x) { return std::tan(x); }, [](float x) { return std::tan(x); });
}

NDArray* neg(const NDArray& a) {
    return apply_unary(a, [](double x) { return -x; }, [](float x) { return -x; });
}

NDArray* equal(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return (x == y) ? 1.0 : 0.0; },
                             [](float x, float y) { return (x == y) ? 1.0f : 0.0f; });
}

NDArray* greater(const NDArray& a, const NDArray& b) {
    return apply_elementwise(a, b, [](double x, double y) { return (x > y) ? 1.0 : 0.0; },
                             [](float x, float y) { return (x > y) ? 1.0f : 0.0f; });
}

}  // namespace np

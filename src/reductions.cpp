#include "np/reductions.h"

#include <cstdint>
#include <vector>

#include "np/allocator.h"

namespace np {

namespace {

template <typename T>
NDArray* reduce_sum(const NDArray& array) {
    NDArray* out = create_array(nullptr, 0, array.dtype);
    if (!out) {
        return nullptr;
    }
    T result = 0;
    std::size_t total = numel(array);
    if (total == 0) {
        return out;
    }
    if (is_contiguous(array)) {
        const T* data = static_cast<const T*>(array.data);
        for (std::size_t i = 0; i < total; ++i) {
            result += data[i];
        }
    } else {
        std::vector<std::size_t> index(array.ndim, 0);
        for (std::size_t linear = 0; linear < total; ++linear) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < array.ndim; ++i) {
                offset += index[i] * array.strides[i];
            }
            const T* value = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(array.data) + offset);
            result += *value;
            for (std::size_t i = array.ndim; i-- > 0;) {
                index[i] += 1;
                if (index[i] < array.shape[i]) {
                    break;
                }
                index[i] = 0;
            }
        }
    }
    *static_cast<T*>(out->data) = result;
    return out;
}

}  // namespace

NDArray* sum(const NDArray& array) {
    if (array.dtype == DType::Float32) {
        return reduce_sum<float>(array);
    }
    if (array.dtype == DType::Float64) {
        return reduce_sum<double>(array);
    }
    return nullptr;
}

NDArray* mean(const NDArray& array) {
    NDArray* total = sum(array);
    if (!total) {
        return nullptr;
    }
    std::size_t count = numel(array);
    if (count == 0) {
        return total;
    }
    if (array.dtype == DType::Float32) {
        *static_cast<float*>(total->data) /= static_cast<float>(count);
    } else if (array.dtype == DType::Float64) {
        *static_cast<double*>(total->data) /= static_cast<double>(count);
    }
    return total;
}

namespace {

template <typename T>
NDArray* reduce_max(const NDArray& array) {
    NDArray* out = create_array(nullptr, 0, array.dtype);
    if (!out) {
        return nullptr;
    }
    std::size_t total = numel(array);
    if (total == 0) {
        return out;
    }
    T result = 0;
    bool initialized = false;
    if (is_contiguous(array)) {
        const T* data = static_cast<const T*>(array.data);
        for (std::size_t i = 0; i < total; ++i) {
            if (!initialized || data[i] > result) {
                result = data[i];
                initialized = true;
            }
        }
    } else {
        std::vector<std::size_t> index(array.ndim, 0);
        for (std::size_t linear = 0; linear < total; ++linear) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < array.ndim; ++i) {
                offset += index[i] * array.strides[i];
            }
            const T* value = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(array.data) + offset);
            if (!initialized || *value > result) {
                result = *value;
                initialized = true;
            }
            for (std::size_t i = array.ndim; i-- > 0;) {
                index[i] += 1;
                if (index[i] < array.shape[i]) {
                    break;
                }
                index[i] = 0;
            }
        }
    }
    *static_cast<T*>(out->data) = result;
    return out;
}

template <typename T>
NDArray* reduce_min(const NDArray& array) {
    NDArray* out = create_array(nullptr, 0, array.dtype);
    if (!out) {
        return nullptr;
    }
    std::size_t total = numel(array);
    if (total == 0) {
        return out;
    }
    T result = 0;
    bool initialized = false;
    if (is_contiguous(array)) {
        const T* data = static_cast<const T*>(array.data);
        for (std::size_t i = 0; i < total; ++i) {
            if (!initialized || data[i] < result) {
                result = data[i];
                initialized = true;
            }
        }
    } else {
        std::vector<std::size_t> index(array.ndim, 0);
        for (std::size_t linear = 0; linear < total; ++linear) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < array.ndim; ++i) {
                offset += index[i] * array.strides[i];
            }
            const T* value = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(array.data) + offset);
            if (!initialized || *value < result) {
                result = *value;
                initialized = true;
            }
            for (std::size_t i = array.ndim; i-- > 0;) {
                index[i] += 1;
                if (index[i] < array.shape[i]) {
                    break;
                }
                index[i] = 0;
            }
        }
    }
    *static_cast<T*>(out->data) = result;
    return out;
}

template <typename T>
std::size_t reduce_argmax_index(const NDArray& array) {
    std::size_t total = numel(array);
    if (total == 0) {
        return 0;
    }
    std::size_t arg = 0;
    if (is_contiguous(array)) {
        const T* data = static_cast<const T*>(array.data);
        T best = data[0];
        for (std::size_t i = 1; i < total; ++i) {
            if (data[i] > best) {
                best = data[i];
                arg = i;
            }
        }
    } else {
        std::vector<std::size_t> index(array.ndim, 0);
        T best = 0;
        bool initialized = false;
        for (std::size_t linear = 0; linear < total; ++linear) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < array.ndim; ++i) {
                offset += index[i] * array.strides[i];
            }
            const T* value = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(array.data) + offset);
            if (!initialized || *value > best) {
                best = *value;
                arg = linear;
                initialized = true;
            }
            for (std::size_t i = array.ndim; i-- > 0;) {
                index[i] += 1;
                if (index[i] < array.shape[i]) {
                    break;
                }
                index[i] = 0;
            }
        }
    }
    return arg;
}

template <typename T>
std::size_t reduce_argmin_index(const NDArray& array) {
    std::size_t total = numel(array);
    if (total == 0) {
        return 0;
    }
    std::size_t arg = 0;
    if (is_contiguous(array)) {
        const T* data = static_cast<const T*>(array.data);
        T best = data[0];
        for (std::size_t i = 1; i < total; ++i) {
            if (data[i] < best) {
                best = data[i];
                arg = i;
            }
        }
    } else {
        std::vector<std::size_t> index(array.ndim, 0);
        T best = 0;
        bool initialized = false;
        for (std::size_t linear = 0; linear < total; ++linear) {
            std::size_t offset = 0;
            for (std::size_t i = 0; i < array.ndim; ++i) {
                offset += index[i] * array.strides[i];
            }
            const T* value = reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(array.data) + offset);
            if (!initialized || *value < best) {
                best = *value;
                arg = linear;
                initialized = true;
            }
            for (std::size_t i = array.ndim; i-- > 0;) {
                index[i] += 1;
                if (index[i] < array.shape[i]) {
                    break;
                }
                index[i] = 0;
            }
        }
    }
    return arg;
}

NDArray* reduce_arg(const NDArray& array, bool is_max) {
    NDArray* out = create_array(nullptr, 0, DType::Float64);
    if (!out) {
        return nullptr;
    }
    std::size_t index = 0;
    if (array.dtype == DType::Float32) {
        index = is_max ? reduce_argmax_index<float>(array) : reduce_argmin_index<float>(array);
    } else if (array.dtype == DType::Float64) {
        index = is_max ? reduce_argmax_index<double>(array) : reduce_argmin_index<double>(array);
    }
    *static_cast<double*>(out->data) = static_cast<double>(index);
    return out;
}

}  // namespace

NDArray* max(const NDArray& array) {
    if (array.dtype == DType::Float32) {
        return reduce_max<float>(array);
    }
    if (array.dtype == DType::Float64) {
        return reduce_max<double>(array);
    }
    return nullptr;
}

NDArray* min(const NDArray& array) {
    if (array.dtype == DType::Float32) {
        return reduce_min<float>(array);
    }
    if (array.dtype == DType::Float64) {
        return reduce_min<double>(array);
    }
    return nullptr;
}

NDArray* argmax(const NDArray& array) {
    return reduce_arg(array, true);
}

NDArray* argmin(const NDArray& array) {
    return reduce_arg(array, false);
}

}  // namespace np

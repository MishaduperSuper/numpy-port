#include "np/ndarray.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "np/allocator.h"

namespace np {

namespace {
std::vector<std::size_t> compute_strides(const std::vector<std::size_t>& shape, std::size_t itemsize) {
    std::vector<std::size_t> strides(shape.size(), 0);
    if (shape.empty()) {
        return strides;
    }
    std::size_t stride = itemsize;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}
}  // namespace

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        default:
            throw std::runtime_error("Unsupported dtype");
    }
}

NDArray* create_array(const std::size_t* shape, std::size_t ndim, DType dtype) {
    NDArray* array = new NDArray();
    array->ndim = ndim;
    array->dtype = dtype;
    if (ndim > 0 && shape) {
        array->shape.assign(shape, shape + ndim);
    } else {
        array->shape.clear();
    }
    array->strides = compute_strides(array->shape, dtype_size(dtype));
    array->owns_data = true;
    std::size_t total = 1;
    for (std::size_t dim : array->shape) {
        total *= dim;
    }
    array->data = np_alloc(total * dtype_size(dtype));
    if (!array->data && total > 0) {
        delete array;
        return nullptr;
    }
    std::size_t bytes = total * dtype_size(dtype);
    if (bytes > 0) {
        std::memset(array->data, 0, bytes);
    }
    return array;
}

NDArray* create_array_like(const NDArray& other) {
    return create_array(other.shape.data(), other.ndim, other.dtype);
}

void free_array(NDArray* array) {
    if (!array) {
        return;
    }
    if (array->owns_data) {
        np_free(array->data);
    }
    delete array;
}

std::size_t numel(const NDArray& array) {
    if (array.shape.empty()) {
        return 1;
    }
    return std::accumulate(array.shape.begin(), array.shape.end(), static_cast<std::size_t>(1),
                           [](std::size_t a, std::size_t b) { return a * b; });
}

bool is_contiguous(const NDArray& array) {
    if (array.ndim == 0) {
        return true;
    }
    std::size_t expected = dtype_size(array.dtype);
    for (std::size_t i = array.ndim; i-- > 0;) {
        if (array.strides[i] != expected) {
            return false;
        }
        expected *= array.shape[i];
    }
    return true;
}

NDArray* reshape(const NDArray& array, const std::size_t* new_shape, std::size_t new_ndim) {
    std::size_t original = numel(array);
    std::size_t requested = 1;
    for (std::size_t i = 0; i < new_ndim; ++i) {
        requested *= new_shape[i];
    }
    if (original != requested || !is_contiguous(array)) {
        return nullptr;
    }
    NDArray* view = new NDArray();
    view->data = array.data;
    view->dtype = array.dtype;
    view->ndim = new_ndim;
    view->shape.assign(new_shape, new_shape + new_ndim);
    view->strides = compute_strides(view->shape, dtype_size(view->dtype));
    view->owns_data = false;
    return view;
}

NDArray* transpose(const NDArray& array) {
    NDArray* view = new NDArray();
    view->data = array.data;
    view->dtype = array.dtype;
    view->ndim = array.ndim;
    view->shape = array.shape;
    view->strides = array.strides;
    std::reverse(view->shape.begin(), view->shape.end());
    std::reverse(view->strides.begin(), view->strides.end());
    view->owns_data = false;
    return view;
}

NDArray* squeeze(const NDArray& array) {
    NDArray* view = new NDArray();
    view->data = array.data;
    view->dtype = array.dtype;
    view->owns_data = false;
    for (std::size_t i = 0; i < array.ndim; ++i) {
        if (array.shape[i] != 1) {
            view->shape.push_back(array.shape[i]);
            view->strides.push_back(array.strides[i]);
        }
    }
    view->ndim = view->shape.size();
    return view;
}

NDArray* expand_dims(const NDArray& array, std::size_t axis) {
    if (axis > array.ndim) {
        return nullptr;
    }
    NDArray* view = new NDArray();
    view->data = array.data;
    view->dtype = array.dtype;
    view->owns_data = false;
    view->shape = array.shape;
    view->strides = array.strides;
    view->shape.insert(view->shape.begin() + static_cast<std::ptrdiff_t>(axis), 1);
    std::size_t stride = dtype_size(array.dtype);
    if (axis < array.ndim) {
        stride = array.strides[axis];
    }
    view->strides.insert(view->strides.begin() + static_cast<std::ptrdiff_t>(axis), stride);
    view->ndim = view->shape.size();
    return view;
}

NDArray* broadcast_to(const NDArray& array, const std::size_t* shape, std::size_t ndim) {
    if (!shape) {
        return nullptr;
    }
    if (ndim < array.ndim) {
        return nullptr;
    }
    NDArray* view = new NDArray();
    view->data = array.data;
    view->dtype = array.dtype;
    view->owns_data = false;
    view->shape.assign(shape, shape + ndim);
    view->strides.assign(ndim, 0);

    std::size_t offset = ndim - array.ndim;
    for (std::size_t i = 0; i < ndim; ++i) {
        std::size_t dim = view->shape[i];
        if (i < offset) {
            view->strides[i] = 0;
            continue;
        }
        std::size_t src_dim = array.shape[i - offset];
        if (src_dim != dim && src_dim != 1) {
            delete view;
            return nullptr;
        }
        view->strides[i] = (src_dim == 1) ? 0 : array.strides[i - offset];
    }
    view->ndim = ndim;
    return view;
}

}  // namespace np

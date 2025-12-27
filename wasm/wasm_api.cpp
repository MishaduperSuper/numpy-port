#include "np/wasm_api.h"

#include "np/kernels.h"
#include "np/linalg.h"
#include "np/ndarray.h"

struct np_array {
    np::NDArray* impl;
};

namespace {

np_array* wrap(np::NDArray* array) {
    if (!array) {
        return nullptr;
    }
    np_array* wrapper = new np_array();
    wrapper->impl = array;
    return wrapper;
}

}  // namespace

extern "C" {

np_array* np_create_array(const std::size_t* shape, std::size_t ndim, int dtype) {
    np::DType dtype_enum = (dtype == 0) ? np::DType::Float32 : np::DType::Float64;
    return wrap(np::create_array(shape, ndim, dtype_enum));
}

void np_free_array(np_array* array) {
    if (!array) {
        return;
    }
    np::free_array(array->impl);
    delete array;
}

np_array* np_add_arrays(const np_array* a, const np_array* b) {
    if (!a || !b) {
        return nullptr;
    }
    return wrap(np::add(*a->impl, *b->impl));
}

np_array* np_dot_arrays(const np_array* a, const np_array* b) {
    if (!a || !b) {
        return nullptr;
    }
    return wrap(np::dot(*a->impl, *b->impl));
}

void* np_array_data(np_array* array) {
    if (!array) {
        return nullptr;
    }
    return array->impl->data;
}

const std::size_t* np_array_shape(const np_array* array) {
    if (!array) {
        return nullptr;
    }
    return array->impl->shape.data();
}

std::size_t np_array_ndim(const np_array* array) {
    if (!array) {
        return 0;
    }
    return array->impl->ndim;
}

int np_array_dtype(const np_array* array) {
    if (!array) {
        return -1;
    }
    return (array->impl->dtype == np::DType::Float32) ? 0 : 1;
}

}  // extern "C"

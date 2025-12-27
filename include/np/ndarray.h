#ifndef NP_NDARRAY_H
#define NP_NDARRAY_H

// NDArray core: shape/strides/dtype metadata and view helpers.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace np {

enum class DType : std::uint8_t {
    Float32 = 0,
    Float64 = 1,
};

std::size_t dtype_size(DType dtype);

struct NDArray {
    void* data;
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;  // in bytes
    std::size_t ndim;
    DType dtype;
    bool owns_data;
};

NDArray* create_array(const std::size_t* shape, std::size_t ndim, DType dtype);
NDArray* create_array_like(const NDArray& other);
void free_array(NDArray* array);

std::size_t numel(const NDArray& array);
bool is_contiguous(const NDArray& array);

NDArray* reshape(const NDArray& array, const std::size_t* new_shape, std::size_t new_ndim);
NDArray* transpose(const NDArray& array);
NDArray* squeeze(const NDArray& array);
NDArray* expand_dims(const NDArray& array, std::size_t axis);
NDArray* broadcast_to(const NDArray& array, const std::size_t* shape, std::size_t ndim);

}  // namespace np

#endif  // NP_NDARRAY_H

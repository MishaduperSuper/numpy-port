#include <cassert>
#include <cmath>
#include <cstddef>

#include "np/kernels.h"
#include "np/ndarray.h"
#include "np/reductions.h"

namespace {

void fill_sequence(np::NDArray* array) {
    std::size_t total = np::numel(*array);
    if (array->dtype == np::DType::Float32) {
        auto* ptr = static_cast<float*>(array->data);
        for (std::size_t i = 0; i < total; ++i) {
            ptr[i] = static_cast<float>(i + 1);
        }
    } else {
        auto* ptr = static_cast<double*>(array->data);
        for (std::size_t i = 0; i < total; ++i) {
            ptr[i] = static_cast<double>(i + 1);
        }
    }
}

}  // namespace

int main() {
    std::size_t shape[] = {2, 2};
    np::NDArray* a = np::create_array(shape, 2, np::DType::Float32);
    np::NDArray* b = np::create_array(shape, 2, np::DType::Float32);
    fill_sequence(a);
    fill_sequence(b);

    np::NDArray* add = np::add(*a, *b);
    auto* add_data = static_cast<float*>(add->data);
    assert(add_data[0] == 2.0f);

    np::NDArray* exp_val = np::exp(*a);
    auto* exp_data = static_cast<float*>(exp_val->data);
    assert(exp_data[0] > 2.7f && exp_data[0] < 2.8f);

    np::NDArray* maximum = np::maximum(*a, *b);
    auto* max_data = static_cast<float*>(maximum->data);
    assert(max_data[3] == 4.0f);

    np::NDArray* cmp = np::greater(*a, *b);
    auto* cmp_data = static_cast<float*>(cmp->data);
    assert(cmp_data[0] == 0.0f);

    np::NDArray* sum = np::sum(*a);
    assert(static_cast<float*>(sum->data)[0] == 10.0f);

    np::NDArray* max_val = np::max(*a);
    assert(static_cast<float*>(max_val->data)[0] == 4.0f);

    np::NDArray* argmax_val = np::argmax(*a);
    assert(static_cast<double*>(argmax_val->data)[0] == 3.0);

    np::NDArray* squeezed = np::squeeze(*a);
    assert(squeezed->ndim == 2);

    np::NDArray* expanded = np::expand_dims(*a, 0);
    assert(expanded->ndim == 3);
    assert(expanded->shape[0] == 1);

    std::size_t bshape[] = {2, 2, 2};
    np::NDArray* broadcasted = np::broadcast_to(*a, bshape, 3);
    assert(broadcasted != nullptr);
    assert(broadcasted->shape[0] == 2);

    np::free_array(a);
    np::free_array(b);
    np::free_array(add);
    np::free_array(exp_val);
    np::free_array(maximum);
    np::free_array(cmp);
    np::free_array(sum);
    np::free_array(max_val);
    np::free_array(argmax_val);
    np::free_array(squeezed);
    np::free_array(expanded);
    np::free_array(broadcasted);

    return 0;
}

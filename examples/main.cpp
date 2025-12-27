#include <iostream>

#include "np/kernels.h"
#include "np/linalg.h"
#include "np/ndarray.h"
#include "np/reductions.h"

int main() {
    std::size_t shape_a[] = {2, 2};
    std::size_t shape_b[] = {2, 2};

    np::NDArray* a = np::create_array(shape_a, 2, np::DType::Float32);
    np::NDArray* b = np::create_array(shape_b, 2, np::DType::Float32);

    float* data_a = static_cast<float*>(a->data);
    float* data_b = static_cast<float*>(b->data);

    data_a[0] = 1.0f; data_a[1] = 2.0f; data_a[2] = 3.0f; data_a[3] = 4.0f;
    data_b[0] = 5.0f; data_b[1] = 6.0f; data_b[2] = 7.0f; data_b[3] = 8.0f;

    np::NDArray* c = np::add(*a, *b);
    np::NDArray* d = np::dot(*a, *b);
    np::NDArray* s = np::sum(*a);

    std::cout << "Add result: " << static_cast<float*>(c->data)[0] << ", "
              << static_cast<float*>(c->data)[1] << ", "
              << static_cast<float*>(c->data)[2] << ", "
              << static_cast<float*>(c->data)[3] << "\n";

    std::cout << "Dot result: " << static_cast<float*>(d->data)[0] << ", "
              << static_cast<float*>(d->data)[1] << ", "
              << static_cast<float*>(d->data)[2] << ", "
              << static_cast<float*>(d->data)[3] << "\n";

    std::cout << "Sum result: " << static_cast<float*>(s->data)[0] << "\n";

    np::free_array(a);
    np::free_array(b);
    np::free_array(c);
    np::free_array(d);
    np::free_array(s);

    return 0;
}

#ifndef NP_KERNELS_H
#define NP_KERNELS_H

// Elementwise math kernels (add/sub/mul/div).
#include "np/ndarray.h"

namespace np {

NDArray* add(const NDArray& a, const NDArray& b);
NDArray* sub(const NDArray& a, const NDArray& b);
NDArray* mul(const NDArray& a, const NDArray& b);
NDArray* div(const NDArray& a, const NDArray& b);
NDArray* maximum(const NDArray& a, const NDArray& b);
NDArray* minimum(const NDArray& a, const NDArray& b);
NDArray* pow(const NDArray& a, const NDArray& b);

NDArray* exp(const NDArray& a);
NDArray* log(const NDArray& a);
NDArray* sqrt(const NDArray& a);
NDArray* abs(const NDArray& a);
NDArray* sin(const NDArray& a);
NDArray* cos(const NDArray& a);
NDArray* tan(const NDArray& a);
NDArray* neg(const NDArray& a);
NDArray* equal(const NDArray& a, const NDArray& b);
NDArray* greater(const NDArray& a, const NDArray& b);

}  // namespace np

#endif  // NP_KERNELS_H

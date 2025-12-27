#ifndef NP_LINALG_H
#define NP_LINALG_H

// Minimal BLAS-like linear algebra routines.
#include "np/ndarray.h"

namespace np {

NDArray* dot(const NDArray& a, const NDArray& b);

}  // namespace np

#endif  // NP_LINALG_H

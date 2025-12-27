#ifndef NP_REDUCTIONS_H
#define NP_REDUCTIONS_H

// Reduction kernels (sum/mean) over all elements.
#include "np/ndarray.h"

namespace np {

NDArray* sum(const NDArray& array);
NDArray* mean(const NDArray& array);
NDArray* max(const NDArray& array);
NDArray* min(const NDArray& array);
NDArray* argmax(const NDArray& array);
NDArray* argmin(const NDArray& array);

}  // namespace np

#endif  // NP_REDUCTIONS_H

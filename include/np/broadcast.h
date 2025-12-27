#ifndef NP_BROADCAST_H
#define NP_BROADCAST_H

// Broadcasting utilities for elementwise kernels.
#include <cstddef>
#include <vector>

#include "np/ndarray.h"

namespace np {

struct BroadcastPlan {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides_a;
    std::vector<std::size_t> strides_b;
};

bool compute_broadcast(const NDArray& a, const NDArray& b, BroadcastPlan* plan);

}  // namespace np

#endif  // NP_BROADCAST_H

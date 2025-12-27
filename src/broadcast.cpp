#include "np/broadcast.h"

#include <algorithm>

namespace np {

bool compute_broadcast(const NDArray& a, const NDArray& b, BroadcastPlan* plan) {
    if (!plan) {
        return false;
    }
    std::size_t ndim = std::max(a.ndim, b.ndim);
    plan->shape.assign(ndim, 1);
    plan->strides_a.assign(ndim, 0);
    plan->strides_b.assign(ndim, 0);

    for (std::size_t i = 0; i < ndim; ++i) {
        std::size_t idx_a = (i < ndim - a.ndim) ? 0 : i - (ndim - a.ndim);
        std::size_t idx_b = (i < ndim - b.ndim) ? 0 : i - (ndim - b.ndim);
        std::size_t dim_a = (i < ndim - a.ndim) ? 1 : a.shape[idx_a];
        std::size_t dim_b = (i < ndim - b.ndim) ? 1 : b.shape[idx_b];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
        plan->shape[i] = std::max(dim_a, dim_b);
        plan->strides_a[i] = (dim_a == 1) ? 0 : a.strides[idx_a];
        plan->strides_b[i] = (dim_b == 1) ? 0 : b.strides[idx_b];
    }
    return true;
}

}  // namespace np

#ifndef NP_ALLOCATOR_H
#define NP_ALLOCATOR_H

// Memory allocator module used by NDArray and kernels.
#include <cstddef>

namespace np {

void* np_alloc(std::size_t bytes);
void np_free(void* ptr);

}  // namespace np

#endif  // NP_ALLOCATOR_H

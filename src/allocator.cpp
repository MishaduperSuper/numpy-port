#include "np/allocator.h"

#include <cstdlib>

namespace np {

void* np_alloc(std::size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    return std::malloc(bytes);
}

void np_free(void* ptr) {
    std::free(ptr);
}

}  // namespace np

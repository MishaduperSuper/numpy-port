#ifndef NP_WASM_API_H
#define NP_WASM_API_H

// C API for WASM bindings and external consumers.
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct np_array np_array;

np_array* np_create_array(const std::size_t* shape, std::size_t ndim, int dtype);
void np_free_array(np_array* array);
np_array* np_add_arrays(const np_array* a, const np_array* b);
np_array* np_dot_arrays(const np_array* a, const np_array* b);

void* np_array_data(np_array* array);
const std::size_t* np_array_shape(const np_array* array);
std::size_t np_array_ndim(const np_array* array);
int np_array_dtype(const np_array* array);

#ifdef __cplusplus
}
#endif

#endif  // NP_WASM_API_H

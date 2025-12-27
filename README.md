# NumPy-Port (WASM-ready)

Минимально жизнеспособная реализация NumPy на C++17 без зависимости от CPython, ориентированная на сборку в WebAssembly через Emscripten. Проект предоставляет ядро ndarray, модуль выделения памяти, math kernels, broadcasting, reductions и минимальный BLAS-like слой для dot.

## Архитектура

- **ndarray core**: структура NDArray, управление shape/strides/dtype, reshape/transpose.
- **memory allocator**: минимальный аллокатор через `malloc/free` с единым API.
- **math kernels**: elementwise операции add/sub/mul/div и расширенный набор unary/binary/compare операций.
- **broadcasting**: отдельный модуль вычисления broadcast формы и stride-планов.
- **reductions**: sum/mean/max/min/argmax/argmin по всем элементам.
- **linear algebra**: минимальный dot для 1D/2D комбинаций.
- **WASM C API**: набор функций `create_array`, `free_array`, `add_arrays`, `dot_arrays` и helpers для JS биндингов.

## Структура проекта

```
/include
  /np
    allocator.h
    broadcast.h
    kernels.h
    linalg.h
    ndarray.h
  reductions.h
  wasm_api.h
/src
  allocator.cpp
  broadcast.cpp
  kernels.cpp
  linalg.cpp
  ndarray.cpp
  reductions.cpp
/wasm
  wasm_api.cpp
/examples
  main.cpp
 /tests
  test_main.cpp
CMakeLists.txt
README.md
```

## Требования

- C++17 компилятор
- CMake 3.16+
- Для WASM: Emscripten SDK

## Сборка (native)

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
./np_example
```

## Как собрать и запустить тесты (native)

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build .
ctest --output-on-failure
```

## Сборка WASM на Windows

### 1) Установка Emscripten

- Установите [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).
- В PowerShell:

```powershell
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
```

### 2) Настройка среды

```powershell
./emsdk_env.ps1
```

Убедитесь, что `emcc` доступен в `PATH`.

### 3) Команды сборки

```powershell
mkdir build-wasm
cd build-wasm
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release
emmake cmake --build .
```

По умолчанию будет собрана библиотека `numpy_port` из `npwasm`.
Для экспорта функций добавьте флаги Emscripten на этапе линковки (пример ниже).

## Как собрать WASM без тестов плюс протестировать код

```powershell
mkdir build-wasm
cd build-wasm
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
emmake cmake --build .
```

Для проверки корректности выполните native-тесты перед сборкой WASM:

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build .
ctest --output-on-failure
```

## Запуск WASM

### Через Node.js

```bash
node examples/wasm_node_example.js
```

### Через браузер

- Поднимите локальный сервер:

```bash
python -m http.server 8000
```

- Откройте `http://localhost:8000/examples/wasm_browser_example.html`.

## Пример использования из JavaScript

```javascript
// Имена экспортов соответствуют C API.
// Пример загрузки через MODULARIZE/EXPORT_NAME приведен для наглядности.

Module.onRuntimeInitialized = () => {
  const shapePtr = Module._malloc(2 * 4);
  Module.setValue(shapePtr, 2, "i32");
  Module.setValue(shapePtr + 4, 2, "i32");

  const a = Module._np_create_array(shapePtr, 2, 0); // dtype 0 = float32
  const b = Module._np_create_array(shapePtr, 2, 0);

  const aData = Module._np_array_data(a);
  const bData = Module._np_array_data(b);

  const f32 = Module.HEAPF32;
  f32[aData / 4 + 0] = 1;
  f32[aData / 4 + 1] = 2;
  f32[aData / 4 + 2] = 3;
  f32[aData / 4 + 3] = 4;

  f32[bData / 4 + 0] = 5;
  f32[bData / 4 + 1] = 6;
  f32[bData / 4 + 2] = 7;
  f32[bData / 4 + 3] = 8;

  const c = Module._np_add_arrays(a, b);
  const cData = Module._np_array_data(c);
  console.log("add:", f32[cData / 4 + 0], f32[cData / 4 + 1]);

  Module._np_free_array(a);
  Module._np_free_array(b);
  Module._np_free_array(c);
  Module._free(shapePtr);
};
```

### Рекомендуемые флаги Emscripten

```bash
emcc -O3 -s MODULARIZE=1 -s EXPORT_NAME=\"Module\" \
  -s EXPORTED_FUNCTIONS=\"['_np_create_array','_np_free_array','_np_add_arrays','_np_dot_arrays','_np_array_data','_np_array_shape','_np_array_ndim','_np_array_dtype']\" \
  -s EXPORTED_RUNTIME_METHODS=\"['setValue','getValue']\" \
  -o numpy_port.js
```

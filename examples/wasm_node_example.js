const fs = require("fs");
const path = require("path");

const wasmPath = path.join(__dirname, "../build-wasm/numpy_port.wasm");
const jsPath = path.join(__dirname, "../build-wasm/numpy_port.js");

if (!fs.existsSync(jsPath)) {
  console.error("Build the WASM target first (see README)." );
  process.exit(1);
}

const ModuleFactory = require(jsPath);

ModuleFactory({
  locateFile: (file) => path.join(__dirname, "../build-wasm", file),
}).then((Module) => {
  const shapePtr = Module._malloc(2 * 4);
  Module.setValue(shapePtr, 2, "i32");
  Module.setValue(shapePtr + 4, 2, "i32");

  const a = Module._np_create_array(shapePtr, 2, 0);
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
});

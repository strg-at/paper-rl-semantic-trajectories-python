from cffi import FFI
from typing import Generator
import numpy as np
import numpy.typing as npt

ffibuilder = FFI()
ffibuilder.cdef(
    """\
typedef double real;

typedef struct cooccur_arr {
  int *words_1;
  int *words_2;
  real *vals;
} COO_ARR;

unsigned long read_cooccur(COO_ARR *coo_arr, FILE *f, int lines);
void free_cooarr(COO_ARR *coo_arr);
"""
)


creclib = ffibuilder.dlopen("glove_ffi/_crec.cpython-312-x86_64-linux-gnu.so")


def cooccurrence_iterator(cooccur_filepath: str, batch_size: int) -> Generator[
    tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.floating]],
    None,
    None,
]:
    with open(cooccur_filepath, "rb") as f:
        while True:
            # The struct will be freed once garbage collected
            coo_arr = ffibuilder.new("COO_ARR*")
            read_lines = creclib.read_cooccur(coo_arr, f, batch_size)
            if read_lines == 0:
                break
            w1_buf = ffibuilder.buffer(coo_arr.words_1, read_lines * ffibuilder.sizeof("int"))
            w2_buf = ffibuilder.buffer(coo_arr.words_2, read_lines * ffibuilder.sizeof("int"))
            vals_buf = ffibuilder.buffer(coo_arr.vals, read_lines * ffibuilder.sizeof("real"))

            # word indices start at 1 in GloVe cooccurr file, see `glove.c`, line 187, 188.
            w1_np = np.frombuffer(w1_buf, dtype=np.int32, count=read_lines) - 1
            w2_np = np.frombuffer(w2_buf, dtype=np.int32, count=read_lines) - 1
            vals_np = np.frombuffer(vals_buf, dtype=np.float64, count=read_lines)
            yield w1_np, w2_np, vals_np
            # This will free the struct arrays
            creclib.free_cooarr(coo_arr)

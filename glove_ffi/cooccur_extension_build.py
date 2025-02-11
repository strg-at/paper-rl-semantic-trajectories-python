from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef(
    """\
typedef double real;

typedef struct cooccur_rec {
  int word1;
  int word2;
  real val;
} CREC;

typedef struct cooccur_arr {
  int *words_1;
  int *words_2;
  real *vals;
} COO_ARR;

CREC *init_crec(int arr_size);

void free_cooarr(COO_ARR *coo_arr);
unsigned long read_cooccur(COO_ARR *coo_arr, FILE *f, int lines);
"""
)

ffibuilder.set_source(
    "_crec",
    """
#include "cooccur.h"
""",
    sources=["cooccur.c"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

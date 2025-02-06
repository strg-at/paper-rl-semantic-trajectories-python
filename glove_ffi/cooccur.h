#ifndef COOCCUR_H
#define COOCCUR_H

#include <stdio.h>

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

#endif
